"""Evaluate the already frozen model on the pre-registered PlantDoc supplement."""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, atomic_json, metrics_from_matrix, sha256, update_matrix, utc_now, write_csv
from .inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--artifact", type=Path, required=True)
    directory = parser.parse_args().artifact; torch.set_num_threads(4)
    freeze = json.loads((directory / "freeze.json").read_text())
    assert (directory / "test_metrics.json").exists(), "Finish the same frozen model's primary evaluation first"
    assert sha256(directory / "model.pt") == freeze["checkpoint_sha256"]
    assert sha256(directory / "external_protocol.json") == freeze["external_protocol_sha256"]
    assert sha256(directory / "external_manifest.csv") == freeze["external_manifest_sha256"]
    if (directory / "external_metrics.json").exists():
        print("Supplement already evaluated; preserve its saved results."); return
    protocol = json.loads((directory / "external_protocol.json").read_text())
    rows = [row for row in csv.DictReader((directory / "external_manifest.csv").open()) if row["status"] == "included"]
    assert len(rows) == protocol["included_images"] == 183
    predictor = Predictor(directory, "cpu"); classes = predictor.classes; mapping = {name:i for i,name in enumerate(classes)}
    matrix = np.zeros((177, 177), np.int64); predictions = []
    for row in rows:
        path = ROOT / row["image_path"]; assert sha256(path) == row["sha256"]
        result = predictor.predict(path)
        target = mapping[row["class_name"]]; predicted = mapping[result["label"]]
        update_matrix(matrix, [target], [predicted])
        predictions.append({**row, "actual_class": row["class_name"], "predicted_class": result["label"],
                            "score": result["score"], "correct": target == predicted})
    metrics = metrics_from_matrix(matrix, classes)
    covered = [row for row in metrics["per_class"] if row["support"]]
    metrics["macro_f1_all_177_classes"] = metrics.pop("macro_f1")
    metrics["macro_f1"] = sum(row["f1"] for row in covered) / len(covered)
    metrics["macro_f1_definition"] = "Mean F1 over the three represented tomato labels; predictions still use all 177 logits"
    metrics["per_class"] = covered
    baseline_matrix = np.zeros((177, 177), np.int64)
    baseline_matrix[:, mapping[protocol["conditional_rule"]["class_name"]]] = matrix.sum(1)
    baseline = metrics_from_matrix(baseline_matrix, classes)
    baseline_covered = [row for row in baseline["per_class"] if row["support"]]
    baseline["macro_f1"] = sum(row["f1"] for row in baseline_covered) / len(baseline_covered)
    metrics.update({"conditional_rule": {key: baseline[key] for key in ["correct", "sample_count", "accuracy", "macro_f1"]},
                    "conditional_rule_definition": protocol["conditional_rule"], "protocol": protocol,
                    "checkpoint_sha256": freeze["checkpoint_sha256"], "evaluated_utc": utc_now()})
    write_csv(directory / "external_predictions.csv", predictions)
    write_csv(directory / "external_per_class.csv", covered)
    atomic_json(directory / "external_metrics.json", metrics)
    artifact = json.loads((directory / "artifact.json").read_text())
    artifact["supplementary_evaluation_complete"] = True
    artifact["files"] = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
                         if p.is_file() and p.name != "artifact.json" and "__pycache__" not in p.parts}
    atomic_json(directory / "artifact.json", artifact)
    print(json.dumps({key:metrics[key] for key in ["correct", "sample_count", "accuracy", "macro_f1", "conditional_rule"]},indent=2))


if __name__ == "__main__": main()
