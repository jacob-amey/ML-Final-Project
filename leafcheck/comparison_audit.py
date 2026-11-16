"""Audit label coverage and prior data exposure without making test predictions."""

import csv
import json
from collections import Counter
from pathlib import Path
import torch
from .common import ROOT, DATA, atomic_json, sha256
from .data import release


def main():
    frozen = release()
    rows = [row for row in csv.DictReader((DATA / "manifest.csv").open()) if row["status"] == "included"]
    exposure = json.loads((DATA / "metadata/previous_exposures.json").read_text())
    train_hashes = set(exposure["sha256"])
    results = {"dataset_fingerprint": frozen["fingerprint"], "test_predictions_computed": False,
               "scope": "Training exposure and label coverage only; performance not compared here.", "models": {}}
    for run_folder in ["leafcheck_small", "leafcheck_starter300"]:
        path = ROOT / "leafcheck-demo/models" / run_folder / "best_model.pt"
        cp = torch.load(path, map_location="cpu", weights_only=True)
        classes = cp["classes"]
        results["models"][run_folder] = {
            "checkpoint_sha256": sha256(path), "classes": classes, "class_count": len(classes),
            "training_images": cp["config"]["train_images"], "historical_protocol": "Earlier three-class pilot; different splits and known related-view concerns",
            "new_test_exact_overlap_with_recorded_experiment_exposures": sum(row["sha256"] in train_hashes for row in rows if row["split"] == "test"),
            "comparison_policy": "Keep earlier scores labelled historical. Three labels and different training budgets do not support an architecture-superiority claim against the 177-class model."}
    sources = {}
    for split in ["train", "val", "test"]:
        path = ROOT / "PlantDataML.mlproj/Data Sources" / f"{split}.json"
        source = json.loads(path.read_text())
        sources[split] = {"data_url": source.get("dataURL"), "record_sha256": sha256(path)}
    overlap = Counter(row["master_split"] for row in rows if row["split"] == "test")
    results["models"]["createml"] = {
        "class_count": 177, "source_records": sources,
        "new_test_images_from_declared_old_training_split": overlap["train"],
        "new_test_images_from_declared_old_validation_split": overlap["val"],
        "new_test_images_from_declared_old_test_split": overlap["test"],
        "exposure_evidence": "Local Create ML data-source records point to the original master train/val/test folders.",
        "comparison_policy": "Do not evaluate or rank Create ML on the new 177-class holdout: 5,832 images come from its declared training split and 1,683 from its validation split. Older saved testing used a different, less strictly grouped protocol."}
    results["valid_full_177_test_comparison"] = ["selected custom CNN", "most-frequent-training-class rule"]
    atomic_json(ROOT / "docs/comparison_protocol.json", results)
    print(json.dumps({"CreateML_old_training_images_in_new_test": overlap["train"],
                      "CreateML_old_validation_images_in_new_test": overlap["val"],
                      "recorded_pilot_exposure_hashes_in_new_test": sum(row["sha256"] in train_hashes for row in rows if row["split"] == "test")}, indent=2))


if __name__ == "__main__": main()
