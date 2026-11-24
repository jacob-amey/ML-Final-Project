"""Evaluate only the frozen original scoring and validation-selected screen on PLDD-UP."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .inference import Predictor
from .reliability_study import ece, measures
from .select_followup import PROTOCOL, signal_from_probs
from .train import device_for
from .upload_study import ARTIFACT, pixels, probabilities, read, rows, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = parser.parse_args()
    freeze = read(args.run / "field_freeze.json")
    if (args.run / "field_metrics.json").exists():
        raise FileExistsError("Preserving already opened field results")
    assert freeze["protocol_sha256"] == sha256(PROTOCOL)
    assert freeze["development_decision_sha256"] == sha256(args.run / "followup_decision.json")
    expected = read(ROOT / "data/manifests/leafcheck_field_v1_release.json")
    directory = ROOT / "data/processed/leafcheck_field_v1"
    local = read(directory / "release.json")
    assert expected == local
    assert freeze["field_manifest_sha256"] == local["manifest_sha256"] == sha256(directory / "manifest.csv")
    assert freeze["checkpoint_sha256"] == sha256(ARTIFACT / "model.pt")
    assert freeze["class_map_sha256"] == sha256(ARTIFACT / "class_to_idx.json")
    assert freeze["preprocessing_sha256"] == sha256(ARTIFACT / "preprocessing.json")
    assert freeze["field_metrics_used_in_selection"] is False

    members = [r for r in rows(directory / "manifest.csv") if r["status"] == "included"]
    assert len({r["image_id"] for r in members}) == len(members)
    assert all(r["split"] == "test" for r in members)
    collection = [{**r, "path": directory / r["path"]} for r in members]
    torch.set_num_threads(4)
    device = device_for(args.device)
    predictor = Predictor(ARTIFACT, device)
    assert predictor.policy["name"] == freeze["inference_policy"]
    y = np.array([int(r["class_id"]) for r in members])
    assert [predictor.classes[i] for i in y] == [r["class_name"] for r in members]
    p = probabilities(predictor.scoring_model, pixels(collection, 224), predictor.spec, device, 24)
    estimates = p.argmax(1)
    conf = p.max(1)
    accept = signal_from_probs(p, freeze["screen"]) >= freeze["threshold"]
    correct = estimates == y
    baseline = measures(correct, np.ones(len(correct), dtype=bool))
    selected = measures(correct, accept)
    independent = score(y, estimates, predictor.classes, np.unique(y))
    counts = []
    for i in np.unique(y):
        mask = y == i
        counts.append({"label": predictor.classes[i], **measures(correct[mask], accept[mask])})
    guard = (selected["correct_and_accepted"] >= baseline["correct_and_accepted"] * .25
             and selected["incorrect_per_upload"] < baseline["incorrect_per_upload"]
             and selected["accepted_accuracy"] is not None
             and selected["accepted_accuracy"] > baseline["accepted_accuracy"])
    detail = [{"image_id": r["image_id"], "path": r["path"], "actual": predictor.classes[actual],
               "predicted": predictor.classes[pred], "model_score": float(s), "accepted": bool(a)}
              for r, actual, pred, s, a in zip(members, y, estimates, conf, accept)]
    write_csv(args.run / "field_predictions.csv", detail)
    result = {"evaluated_utc": utc_now(), "freeze_sha256": sha256(args.run / "field_freeze.json"),
              "field_manifest_sha256": freeze["field_manifest_sha256"], "device": str(device),
              "full_accuracy": independent["accuracy"], "macro_f1_represented_classes": independent["macro_f1"],
              "macro_recall_represented_classes": float(np.mean([independent["per_class"][i]["recall"] for i in np.unique(y)])),
              "top3_accuracy": float(np.any(np.argsort(p, axis=1)[:, -3:] == y[:, None], 1).mean()),
              "ece15": ece(conf, correct), "blight": independent["blight"],
              "wrong_crop": independent["wrong_crop"], "major_errors": independent["largest_errors"],
              "full_score_metrics": independent, "accept_all": baseline, "screen": selected, "per_class_screen": counts,
              "passes_external_deployment_guard": bool(guard),
              "correct_abstained": int((correct & ~accept).sum()),
              "incorrect_abstained": int((~correct & ~accept).sum()),
              "high_score_errors_at_least_0_9": int((~correct & (conf >= .9)).sum())}
    atomic_json(args.run / "field_metrics.json", result)
    print("FIELD", baseline, selected, "guard", guard, flush=True)


if __name__ == "__main__":
    main()
