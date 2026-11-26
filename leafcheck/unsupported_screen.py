"""Audit unsupported uploads under the deployed rule and fixed score signals."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .field_expansion import load_inputs, prepare_development
from .inference import Predictor
from .potato_policy import distribution
from .upload_study import ARTIFACT, pixels, read, rows


PLAN = ROOT / "configs/leafcheck_unsupported_screen_v1.json"
POLICY = ROOT / "leafcheck-demo/backend/reliability_policy.json"
DATA = ROOT / "data/processed"
MANIFEST = ROOT / "data/manifests/leafcheck_unsupported_openimages_v1_manifest.csv"
RAW = ROOT / "data/raw/leafcheck_field_v8/openimages/images"


def auroc(pos, neg):
    differences = pos[:, None] - neg
    return float(np.mean((differences > 0) + 0.5 * (differences == 0)))


def signals(probabilities, crop_membership):
    highest = np.sort(probabilities, axis=1)[:, -2:]
    return {
        "max_softmax": highest[:, 1],
        "margin": highest[:, 1] - highest[:, 0],
        "negative_entropy": (probabilities * np.log(np.maximum(probabilities, 1e-30))).sum(1),
        "max_crop_mass": (probabilities @ crop_membership).max(1),
    }


def run(output):
    if (output / "metrics.json").exists() or (output / "predictions.csv").exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    plan, policy = read(PLAN), read(POLICY)
    model = Predictor(ARTIFACT, "cpu")
    if model.metadata["checkpoint_sha256"] != plan["checkpoint_sha256"] or policy["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("Expected the deployed model and uncertainty rule")
    classes = model.classes
    _, indexes = np.unique([name.split("__", 1)[0] for name in classes], return_inverse=True)
    membership = np.eye(indexes.max() + 1, dtype=np.float32)[indexes]
    lineage = {"initial_weights": plan["checkpoint_sha256"]}
    _, field, original, supplements = load_inputs(DATA / "leafcheck_field_v2_sampled", lineage)
    _, pk, _, pakistan = load_inputs(DATA / "leafcheck_pakistan_tomato_dev_v1", lineage)
    spec = model.spec
    size = spec["input_size"]
    validation = prepare_development(original, supplements, size)
    validation["pakistan"] = (pixels(pakistan, size), np.array([int(r["class_id"]) for r in pakistan]))
    known = {}
    for name in plan["supported_development"]:
        data, truth = validation[name]
        p = distribution(model.scoring_model, data, spec, model.device, 64)
        known[name] = (signals(p, membership), truth == p.argmax(1))
    positive = {key: np.concatenate([values[0][key] for values in known.values()])
                for key in known[plan["supported_development"][0]][0]}
    negative = [dict(r, path=RAW / (r["image_id"] + ".jpg"))
                for r in rows(MANIFEST) if r["status"] == "included"]
    if (any(r["split"] != "val" for r in negative)
            or len({r["sha256"] for r in negative}) != len(negative)):
        raise ValueError("Unsupported development membership changed")
    raw = pixels(negative, size)
    p = distribution(model.scoring_model, raw, spec, model.device, 64)
    scores = signals(p, membership)
    groups = np.array([row["group"] for row in negative])
    threshold = policy["threshold"]
    accepted = scores["max_softmax"] >= threshold
    per_group = {}
    for key in plan["unsupported_groups"]:
        idx = groups == key
        if not idx.any():
            raise ValueError(f"No photos for {key}")
        per_group[key] = {"count": int(idx.sum()), "accepted": int((accepted & idx).sum()),
                          "auroc_supported_vs_group": {metric: auroc(positive[metric], value[idx])
                                                       for metric, value in scores.items()}}
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.csv"
    write_csv(prediction_path, [
        {"image_id": row["image_id"], "group": row["group"], "category": row["category"],
         "label": classes[j], "accepted": bool(accepted[i]), **{k: float(v[i]) for k, v in scores.items()}}
        for i, (row, j) in enumerate(zip(negative, p.argmax(1), strict=True))
    ])
    atomic_json(output / "metrics.json", {
        "completed_utc": utc_now(), "plan_sha256": sha256(PLAN), "model_sha256": plan["checkpoint_sha256"],
        "manifest_sha256": sha256(MANIFEST), "predictions_sha256": sha256(prediction_path), "threshold": threshold,
        "supported_fingerprints": [field["fingerprint"], pk["fingerprint"]],
        "supported": {name: {"count": len(truth), "accepted": int((values["max_softmax"] >= threshold).sum()),
                             "accepted_correct": int(((values["max_softmax"] >= threshold) & truth).sum())}
                      for name, (values, truth) in known.items()},
        "unsupported": per_group, "test_opened": False, "policy_changed": False,
    })
    print(per_group, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
