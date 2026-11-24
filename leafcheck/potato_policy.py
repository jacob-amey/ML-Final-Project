"""Screen known-potato and full-label policies on validation, then evaluate a frozen release."""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .data import PixelCache
from .field_expansion import load_inputs, prepare_development
from .inference import Predictor
from .potato_study import FIELD_CONFIG, V2, tied_metrics
from .preprocessing import prepare_batch
from .scoring import PredictionScores
from .train import device_for
from .upload_study import ARTIFACT, load_study, pixels, read, rows, score

PLAN = ROOT / "configs/leafcheck_potato_v1_policy.json"
OKARA = ROOT / "data/processed/leafcheck_okara_v1"


@torch.inference_mode()
def distribution(model, raw, spec, device, size=128, columns=None):
    model.eval()
    batches = []
    for batch in raw.split(size):
        log_scores = model(prepare_batch(batch, spec, device))
        if columns is not None:
            log_scores = log_scores[:, columns]
        batches.append(log_scores.softmax(1).cpu())
    return torch.cat(batches).numpy()


@torch.inference_mode()
def master_probabilities(predictor, cache, size=128):
    chunk = []
    for start in range(0, len(cache), size):
        ix = np.arange(start, min(start + size, len(cache)))
        raw, _ = cache.batch(ix)
        chunk.append(distribution(predictor.scoring_model, raw, predictor.spec, predictor.device, size))
    return np.concatenate(chunk)


def metrics(labels, probabilities, classes, represented=None):
    result = score(labels, probabilities.argmax(1), classes, represented)
    return {name: result[name] for name in ("sample_count", "correct", "macro_f1", "accuracy", "wrong_crop",
                                           "blight", "largest_errors", "per_class")}


def load_specialist(entry, config, device):
    from .upload_study import network

    path = Path(entry["checkpoint"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("Specialist checkpoint changed")
    state = torch.load(path, weights_only=True, map_location="cpu")
    if state["config"] != config:
        raise ValueError("Specialist configuration changed")
    model = network(config, device)
    model.load_state_dict(state["weights"], strict=True)
    return PredictionScores(model, {"name": "d4_probability_mean"}).eval()


def second_member(plan, reference, device):
    path = ROOT / plan["member"]
    if sha256(path) != plan["member_sha256"]:
        raise ValueError("Secondary seed checkpoint changed")
    model = Predictor(path, device)
    if model.classes != reference.classes or model.spec != reference.spec:
        raise ValueError("Secondary seed uses a different image contract")
    model.scoring_model = PredictionScores(model.model, reference.policy).eval()
    return model


def describe_set(records, labels):
    return np.array([labels.index(row["class_name"]) for row in records])


def combine(a, b, fraction):
    return a * (1.0 - fraction) + b * fraction


def compact(result):
    return {k: v for k, v in result.items() if k != "per_class"}


def screen(output, training_path, plan_path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = read(plan_path)
    parent = read(training_path)
    config = parent["run"]["config"]
    spec = parent["run"]["spec"]
    classes = config["labels"]
    device = device_for("auto")
    torch.set_num_threads(4)
    frozen, release, previous, new = load_inputs(V2, read(FIELD_CONFIG))
    if (frozen["fingerprint"], release["fingerprint"]) != (parent["run"]["master_fingerprint"],
                                                          parent["run"]["field_fingerprint"]):
        raise ValueError("Training evidence references other source data")
    _, master, _ = load_study(read(ROOT / "configs/upload_blight_v1.json"))
    original = Predictor(ARTIFACT, device)
    if original.metadata["checkpoint_sha256"] != parent["run"]["base_checkpoint"]:
        raise ValueError("Reference weights changed")
    columns = [original.classes.index(name) for name in classes]
    dev = {
        "master": [r for r in master if r["split"] == "val" and r["class_name"] in classes],
        "field": [r for r in new if r["split"] == "val" and r["source"] == "mduma" and r["class_name"] in classes],
    }
    truth = {name: describe_set(records, classes) for name, records in dev.items()}
    masked = {name: distribution(original.scoring_model, pixels(records, original.spec["input_size"]),
                                 original.spec, device, columns=columns) for name, records in dev.items()}
    baseline = {name: metrics(truth[name], p, classes) for name, p in masked.items()}
    encoded = {name: pixels(records, spec["input_size"]) for name, records in dev.items()}
    cases = {}
    thresholds = config["selection"]
    for arm, entries in parent["arms"].items():
        cases[arm] = {}
        for entry in entries:
            specialist = load_specialist(entry, config, device)
            field_probs = {name: distribution(specialist, raw, spec, device) for name, raw in encoded.items()}
            for fraction in plan["fractions"]:
                group = cases[arm].setdefault(str(fraction), [])
                measures = {name: metrics(truth[name], combine(masked[name], field_probs[name], fraction), classes)
                            for name in dev}
                group.append({**{key: entry[key] for key in ("checkpoint", "sha256", "epoch", "seed")},
                              "master_passed": tied_metrics(measures, config, baseline),
                              "master": compact(measures["master"]), "field": compact(measures["field"])})
    candidates = []
    for fraction, entries in cases["field"].items():
        field = np.array([x["field"]["macro_f1"] for x in entries])
        control = np.array([x["field"]["macro_f1"] for x in cases["control"][fraction]])
        gain = float((field - control).mean())
        eligible = (all(x["master_passed"] for x in entries) and (field > control).all()
                    and gain >= thresholds["field_min_gain_average_vs_control"]
                    and field.mean() >= baseline["field"]["macro_f1"] + thresholds["field_min_gain_average_vs_masked"])
        candidates.append({"fraction": float(fraction), "eligible": bool(eligible), "field_mean": float(field.mean()),
                           "control_mean": float(control.mean()), "paired_gain": gain})
    winner = max(candidates, key=lambda r: (r["eligible"], r["field_mean"], -r["fraction"]))
    entries = cases["field"][str(winner["fraction"])]
    selected = max(entries, key=lambda r: r["field"]["macro_f1"])
    known = {"screened": cases, "selection": candidates, "winner": winner,
             "reference": {k: compact(v) for k, v in baseline.items()}, "selected_checkpoint": selected}
    atomic_json(output / "known_potato.json", known)
    print("Known potato", winner, selected["seed"], flush=True)
    auto_plan = plan["automatic_ensemble"]
    sibling = second_member(auto_plan, original, device)
    dataset = PixelCache("val", original.spec["input_size"], frozen)
    started = time.monotonic()
    ref_prob = master_probabilities(original, dataset)
    alt_prob = master_probabilities(sibling, dataset)
    classes_all = original.classes
    base = metrics(dataset.labels, ref_prob, classes_all)
    ensemble = metrics(dataset.labels, combine(ref_prob, alt_prob, .5), classes_all)
    field_refs, field_mixes = {}, {}
    for name, (raw, labels) in prepare_development(previous, new, original.spec["input_size"]).items():
        if name == "banu":
            continue
        ref = distribution(original.scoring_model, raw, original.spec, device)
        alt = distribution(sibling.scoring_model, raw, sibling.spec, device)
        field_refs[name] = compact(metrics(labels, ref, classes_all, np.unique(labels)))
        field_mixes[name] = compact(metrics(labels, combine(ref, alt, .5), classes_all, np.unique(labels)))
    gains = [field_mixes[name]["macro_f1"] - field_refs[name]["macro_f1"] for name in field_refs]
    auto_ok = (ensemble["macro_f1"] >= base["macro_f1"] + auto_plan["master_min_gain"]
               and ensemble["accuracy"] >= base["accuracy"] - auto_plan["master_max_accuracy_drop"]
               and np.mean(gains) >= -auto_plan["field_max_mean_f1_drop"])
    automatic = {"eligible_on_validation": bool(auto_ok), "reference": base, "ensemble": ensemble,
                 "field_reference": field_refs, "field_ensemble": field_mixes,
                 "elapsed_seconds": time.monotonic() - started}
    print("Automatic ensemble", ensemble["macro_f1"], ensemble["accuracy"], "eligible", auto_ok, flush=True)
    atomic_json(output / "decision.json", {
        "training_sha256": sha256(training_path), "plan": plan, "plan_sha256": sha256(plan_path),
        "master_fingerprint": frozen["fingerprint"], "development_utc": utc_now(),
        "reference_sha256": original.metadata["checkpoint_sha256"],
        "class_map_sha256": sha256(ARTIFACT / "class_to_idx.json"),
        "preprocessing_sha256": sha256(ARTIFACT / "preprocessing.json"),
        "specialist_config": config, "specialist_preprocessing": spec, "known_potato": known,
        "automatic": automatic,
    })


def freeze(output, study, release):
    if (output / "freeze.json").exists():
        raise FileExistsError("This study already has a frozen selection")
    chosen = read(study)
    holdout = read(release / "release.json")
    assert sha256(release / "manifest.csv") == holdout["files_sha256"]["manifest.csv"]
    assert chosen["master_fingerprint"] == holdout["master_fingerprint"]
    weights = chosen["known_potato"]["selected_checkpoint"]
    assert sha256(Path(weights["checkpoint"])) == weights["sha256"]
    output.mkdir(parents=True, exist_ok=True)
    automatic = chosen["plan"]["automatic_ensemble"] if chosen["automatic"]["eligible_on_validation"] else None
    atomic_json(output / "freeze.json", {
        "decision_sha256": sha256(study), "manifest_sha256": holdout["files_sha256"]["manifest.csv"],
        "reference_sha256": chosen["reference_sha256"], "class_map_sha256": chosen["class_map_sha256"],
        "preprocessing_sha256": chosen["preprocessing_sha256"], "frozen_utc": utc_now(),
        "specialist": weights, "config": chosen["specialist_config"], "spec": chosen["specialist_preprocessing"],
        "fraction": chosen["known_potato"]["winner"]["fraction"], "automatic": automatic,
        "known_potato_passed_development": chosen["known_potato"]["winner"]["eligible"],
        "prediction_policy": "Original eight-view scoring, then known-potato renormalization. Blend with the scratch CNN using the frozen fraction.",
        "test_purpose": "Evaluate predictions on the previously unopened Okara mirror. Every method uses the same IDs. The known-crop methods assume the user is right about the crop.",
    })


def evaluate(output, release):
    if (output / "metrics.json").exists():
        raise FileExistsError("Preserving the completed test")
    lock = read(output / "freeze.json")
    assert sha256(release / "manifest.csv") == lock["manifest_sha256"]
    assert sha256(ARTIFACT / "class_to_idx.json") == lock["class_map_sha256"]
    assert sha256(ARTIFACT / "preprocessing.json") == lock["preprocessing_sha256"]
    device = device_for("auto")
    torch.set_num_threads(4)
    original = Predictor(ARTIFACT, device)
    assert original.metadata["checkpoint_sha256"] == lock["reference_sha256"]
    entries = [dict(r, path=release / r["output_relative_path"]) for r in rows(release / "manifest.csv")
               if r["status"] == "included" and r["split"] == "test"]
    assert len(entries) == len({r["image_id"] for r in entries})
    classes = lock["config"]["labels"]
    y = describe_set(entries, original.classes)
    represented = [original.classes.index(name) for name in classes]
    raw = pixels(entries, original.spec["input_size"])
    p = distribution(original.scoring_model, raw, original.spec, device)
    masked = p[:, represented] / p[:, represented].sum(1, keepdims=True)
    specialist = load_specialist(lock["specialist"], lock["config"], device)
    q = distribution(specialist, pixels(entries, lock["spec"]["input_size"]), lock["spec"], device)
    predictions = {"automatic_original": p.argmax(1), "masked_original": np.array(represented)[masked.argmax(1)],
                   "field_scratch": np.array(represented)[q.argmax(1)],
                   "known_potato_blend": np.array(represented)[combine(masked, q, lock["fraction"]).argmax(1)]}
    if lock["automatic"]:
        alt = second_member(lock["automatic"], original, device)
        partner = distribution(alt.scoring_model, raw, alt.spec, device)
        predictions["automatic_two_seed"] = combine(p, partner, .5).argmax(1)
    _, master, _ = load_study(read(ROOT / "configs/upload_blight_v1.json"))
    train_counts = Counter(r["class_name"] for r in master if r["split"] == "train")
    most_common = max(represented, key=lambda index: (train_counts[original.classes[index]], -index))
    predictions["most_common_potato"] = np.full(len(entries), most_common, dtype=np.int64)
    evidence = {name: compact(score(y, p, original.classes, represented)) for name, p in predictions.items()}
    records = [dict(image_id=r["image_id"], reference=r["class_name"], path=r["output_relative_path"],
                    **{name: original.classes[p[i]] for name, p in predictions.items()}) for i, r in enumerate(entries)]
    write_csv(output / "predictions.csv", records)
    atomic_json(output / "metrics.json", {"methods": evidence, "evaluated_utc": utc_now(),
                                          "freeze_sha256": sha256(output / "freeze.json"),
                                          "predictions_sha256": sha256(output / "predictions.csv")})
    print({name: (m["correct"], m["sample_count"], m["macro_f1"], m["wrong_crop"])
           for name, m in evidence.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("screen", "freeze", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--release", type=Path, default=OKARA)
    args = parser.parse_args()
    if args.action == "screen":
        if not args.training:
            parser.error("--training is required for screen")
        screen(args.output, args.training, args.plan)
    elif args.action == "freeze":
        if not args.study:
            parser.error("--study is required for freeze")
        freeze(args.output, args.study, args.release)
    else:
        evaluate(args.output, args.release)
