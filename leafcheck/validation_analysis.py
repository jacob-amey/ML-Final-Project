"""Validation-only rare-class/provenance diagnostics; no test access or predictions."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import DATA, atomic_json, metrics_from_matrix, sha256, update_matrix


def analyze(run):
    run = Path(run)
    metrics = json.loads((run / "validation_metrics.json").read_text())
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "complete", "Use a completed procedure so the paired best-model records cannot change during review"
    assert abs(status["best"]["macro_f1"] - metrics["macro_f1"]) < 1e-12
    predictions = list(csv.DictReader((run / "validation_predictions.csv").open()))
    assert len(predictions) == 16836
    metadata = {row["image_id"]: row for row in csv.DictReader((DATA / "manifest.csv").open())
                if row["status"] == "included" and row["split"] == "val"}
    classes = [row["class_name"] for row in metrics["per_class"]]
    bins = defaultdict(list)
    for row in metrics["per_class"]:
        key = "under_20_validation_images" if row["support"] < 20 else "20_to_99_validation_images" if row["support"] < 100 else "at_least_100_validation_images"
        bins[key].append(row)
    support_summary = {key: {"classes": len(rows), "mean_class_f1": sum(row["f1"] for row in rows) / len(rows),
                             "zero_recall_classes": sum(row["recall"] == 0 for row in rows)} for key, rows in bins.items()}
    matrices = defaultdict(lambda: np.zeros((177, 177), np.int64))
    for prediction in predictions:
        row = metadata[prediction["image_id"]]
        key = "PlantDoc_byte_matches" if row["plantdoc_source_paths"] else "documented_PlantVillage_leaf_ID" if row["documented_leaf_ids"] else "other_traceable_images"
        update_matrix(matrices[key], [int(prediction["actual_index"])], [int(prediction["predicted_index"])])
    combined = metrics_from_matrix(sum(matrices.values()), classes)
    assert combined["correct"] == metrics["correct"] and abs(combined["macro_f1"] - metrics["macro_f1"]) < 1e-12
    provenance = {}
    for key, matrix in matrices.items():
        result = metrics_from_matrix(matrix, classes)
        supported = [row for row in result["per_class"] if row["support"]]
        provenance[key] = {"images": result["sample_count"], "correct": result["correct"], "accuracy": result["accuracy"],
                           "covered_classes": len(supported), "macro_f1_over_covered_classes": sum(row["f1"] for row in supported) / len(supported)}
    result = {"checkpoint_selection": "validation macro-F1", "checkpoint_sha256":sha256(run / "best_model.pt"),
              "selected_epoch":status["best"]["epoch"],"validation_macro_f1":metrics["macro_f1"], "support_bins": support_summary,
              "provenance": provenance, "caveat": "These validation strata differ in class composition and share development sources. They are not an independent field benchmark or proof of a causal background shortcut."}
    atomic_json(run / "validation_diagnostics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("run", type=Path)
    analyze(parser.parse_args().run)
