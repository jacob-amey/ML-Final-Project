"""Fit a calibration and abstention policy with no field-test data access."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256
from .inference import Predictor
from .refit_study import CONFIG, load_features, predict_head
from .upload_study import ARTIFACT, load_study, read

PROTOCOL = ROOT / "configs/leafcheck_reliability_v1.json"


def feature_statistics(features, records, class_count):
    group_index = {}
    index = []
    targets = []
    for r in records:
        label = int(r["class_id"])
        key = (r["group_id"], label)
        if key not in group_index:
            group_index[key] = len(targets)
            targets.append(label)
        index.append(group_index[key])
    index = np.array(index)
    targets = np.array(targets)
    grouped = np.zeros((len(targets), features.shape[1]), dtype=np.float64)
    np.add.at(grouped, index, features)
    grouped /= np.bincount(index)[:, None]
    centers = np.stack([grouped[targets == i].mean(0) for i in range(class_count)])
    residual = grouped - centers[targets]
    variance = residual.var(0)
    # A variance floor keeps quiet units from dominating distance.
    scale = np.sqrt(np.maximum(variance, np.median(variance) * .10))
    assert (scale > 0).all()
    return centers.astype(np.float32), scale.astype(np.float32)


def distances(x, centers, scale):
    x = x / scale
    means = centers / scale
    parts = []
    for batch in np.array_split(x, max(1, (len(x) + 1023) // 1024)):
        distance = (batch ** 2).sum(1, keepdims=True) + (means ** 2).sum(1) - 2 * (batch @ means.T)
        parts.append(-distance.min(1) / x.shape[1])
    return np.concatenate(parts)


def measures(correct, accept):
    return {"support": len(correct), "correct_before_screen": int(correct.sum()),
            "accepted": int(accept.sum()), "correct_and_accepted": int((correct & accept).sum()),
            "incorrect_and_accepted": int((~correct & accept).sum()),
            "coverage": float(accept.mean()),
            "accepted_accuracy": float(correct[accept].mean()) if accept.any() else None,
            "incorrect_per_upload": float((~correct & accept).mean())}


def ece(p, correct):
    total = len(p)
    count = 0.0
    bins = np.minimum((p * 15).astype(int), 14)
    for i in range(15):
        ix = bins == i
        if ix.any():
            count += ix.sum() * abs(p[ix].mean() - correct[ix].mean())
    return float(count / total)


def z_transform(signal, calibration):
    values = signal[calibration]
    median = np.median(values)
    width = np.quantile(values, .75) - np.quantile(values, .25)
    assert width > 0
    return (signal - median) / width, {"median": float(median), "iqr": float(width)}


def softmax(scores):
    weights = np.exp(scores - scores.max(1, keepdims=True))
    return weights / weights.sum(1, keepdims=True)


def fit_temperature(log_p, labels):
    target = log_p[np.arange(len(labels)), labels]
    # One-dimensional validation search. Refine around the best grid point.
    lo, hi = -2., 2.
    for _ in range(3):
        probes = np.linspace(lo, hi, 21)
        errors = []
        for t in np.exp(probes):
            scaled = log_p / t
            peak = scaled.max(1)
            nll = peak + np.log(np.exp(scaled - peak[:, None]).sum(1)) - target / t
            errors.append(nll.mean())
        i = int(np.argmin(errors))
        lo, hi = probes[max(0, i - 1)], probes[min(i + 1, len(probes) - 1)]
    return float(np.exp(probes[i]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--head", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = read(CONFIG)
    protocol = read(PROTOCOL)
    frozen, _, _ = load_study(read(ROOT / cfg["base_config"]))
    feature_sets = load_features(args.output, cfg, frozen)
    train, records = feature_sets["train"]
    val, validation = feature_sets["val"]
    fit = np.array([row["master"] == "True" for row in records])
    centers, scales = feature_statistics(train[0, fit].numpy(),
                                         [r for r in records if r["master"] == "True"], 177)
    predictor = Predictor(ARTIFACT)
    head = predictor.model.classifier[-1]
    if args.head:
        artifact = torch.load(args.head, weights_only=True, map_location="cpu")
        assert artifact["identity"]["base"] == cfg["base_checkpoint"]
        head.load_state_dict(artifact["state_dict"])
    p = predict_head(head, val, "cpu")
    labels = np.array([int(r["class_id"]) for r in validation])
    pred = p.argmax(1)
    correct = pred == labels
    master = np.array([r["master"] == "True" for r in validation])
    belongs = np.array([int(hashlib.sha256(("reliability-v1:" + r["group_id"]).encode()).hexdigest()[:8], 16)
                        % 100 < 40 for r in validation])
    calibration = master & belongs
    check = master & ~belongs
    sources = np.array([r["source"] for r in validation])
    supplement = ~master & (sources != "bugwood")
    field = sources == "bugwood"
    log_p = np.log(p.astype(np.float64).clip(min=1e-300))
    t = fit_temperature(log_p[calibration], labels[calibration])
    after = softmax(log_p / t)
    assert np.array_equal(after.argmax(1), pred)
    confidence = p.max(1)
    distance = distances(val[0].numpy(), centers, scales)
    max_z, max_params = z_transform(confidence, calibration)
    dist_z, dist_params = z_transform(distance, calibration)
    groups = {"check_master": check, "supplement": supplement, "field_diagnostic": field}
    signals = {"base_always": np.ones(len(p)), "max_softmax": confidence,
               "feature_distance": distance, "combined": np.minimum(max_z, dist_z)}
    candidates = {}
    for name, signal in signals.items():
        cutoff = float(np.quantile(signal[calibration], 1 - protocol["target_calibration_coverage"]))
        accept = signal >= cutoff
        candidates[name] = {"threshold": cutoff,
                            **{key: measures(correct[ix], accept[ix]) for key, ix in groups.items()}}
    raw = candidates["base_always"]["check_master"]
    simple = candidates["max_softmax"]["supplement"]["coverage"]
    eligible = [name for name in ("feature_distance", "combined")
                if candidates[name]["check_master"]["coverage"] >= protocol["minimum_check_coverage"]
                and candidates[name]["check_master"]["accepted_accuracy"] - raw["accepted_accuracy"] >= protocol["minimum_check_accuracy_gain"]
                and candidates[name]["supplement"]["coverage"] <= protocol["maximum_supplement_coverage"]
                and simple - candidates[name]["supplement"]["coverage"] >= protocol["minimum_supplement_gain_over_confidence"]]
    # Prefer the lower rate of incorrect suggestions, then higher master coverage.
    proposed = min(eligible, key=lambda name: (candidates[name]["check_master"]["incorrect_per_upload"],
                                              -candidates[name]["check_master"]["coverage"])) if eligible else None
    result = {"protocol_sha256": sha256(PROTOCOL), "base_checkpoint": cfg["base_checkpoint"],
              "head_sha256": sha256(args.head) if args.head else None,
              "calibration_groups": len({r["group_id"] for r, belongs in zip(validation, calibration) if belongs}),
              "candidate": proposed, "parameters": {"max_softmax": max_params, "feature_distance": dist_params,
                              "centers": centers.tolist(), "scale": scales.tolist()},
              "references": candidates,
              "temperature": t, "calibration_check": {
                  "top_3_accuracy": float(np.any(np.argsort(p[check], axis=1)[:, -3:] == labels[check, None], 1).mean()),
                  "ece15_before": ece(confidence[check], correct[check]),
                  "ece15_after": ece(after.max(1)[check], correct[check]),
                  "ece15_supplement_before": ece(confidence[supplement], correct[supplement]),
                  "ece15_supplement_after": ece(after.max(1)[supplement], correct[supplement]),
              }}
    atomic_json(args.output / "reliability_development.json", result)
    for name, value in candidates.items():
        print(name, value, flush=True)
    print("SCREEN", proposed, result["calibration_check"], flush=True)


if __name__ == "__main__":
    main()
