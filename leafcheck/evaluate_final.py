"""One final test evaluation after an explicit immutable model/data freeze."""

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from .common import ROOT, atomic_json, environment, hardware, metrics_from_matrix, sha256, utc_now, write_csv
from .data import PixelCache, build_cache, release
from .inference import Predictor
from .train import evaluate
from .report import evaluation_figures, learning_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(); directory = args.artifact
    torch.set_num_threads(4)
    freeze = json.loads((directory / "freeze.json").read_text()); frozen = release()
    assert frozen["fingerprint"] == freeze["dataset_fingerprint"]
    assert sha256(directory / "model.pt") == freeze["checkpoint_sha256"]
    assert sha256(directory / "preprocessing.json") == freeze["preprocessing_sha256"]
    assert sha256(directory / "class_to_idx.json") == freeze["class_mapping_sha256"]
    artifact_record = json.loads((directory / "artifact.json").read_text())
    if artifact_record.get("final_evaluation_complete"):
        print("This unchanged frozen artifact has already been evaluated. Preserving saved test results.")
        print((directory / "test_metrics.json").read_text()[:400]); return
    predictor = Predictor(directory, freeze["evaluation_device"])
    atomic_json(directory / "evaluation_environment.json", {**environment(predictor.device), "hardware":hardware()})
    # Test cache construction is intentionally unavailable in the development CLI.
    build_cache("test", predictor.spec["input_size"], frozen)
    test = PixelCache("test", predictor.spec["input_size"], frozen, allow_test=True)
    assert len(test) == 8425
    if (directory / "test_metrics.json").exists():
        metrics = json.loads((directory / "test_metrics.json").read_text())
        assert metrics["checkpoint_sha256"] == freeze["checkpoint_sha256"]
        predictions = list(csv.DictReader((directory / "test_predictions.csv").open()))
        for row in predictions:
            row["actual_index"] = int(row["actual_index"]); row["predicted_index"] = int(row["predicted_index"])
            row["score"] = float(row["score"]); row["correct"] = row["correct"] == "True"
        print("Resuming figures and packaging from saved predictions; no new test pass.", flush=True)
    else:
        started = time.perf_counter()
        metrics, predictions = evaluate(predictor.model, test, predictor.spec, predictor.device,
                                        frozen["classes"], freeze["evaluation_batch_size"], True)
        metrics.update({"selected_epoch": freeze["selected_epoch"], "checkpoint_sha256": freeze["checkpoint_sha256"],
                        "dataset_fingerprint": frozen["fingerprint"], "evaluation_device": str(predictor.device),
                        "evaluation_batch_size": freeze["evaluation_batch_size"], "evaluated_utc": utc_now(),
                        "evaluation_seconds": time.perf_counter() - started})
        write_csv(directory / "test_predictions.csv", predictions)
        atomic_json(directory / "test_metrics.json", metrics)
    # Independently reconcile saved-prediction counts and class F1 denominators.
    assert sum(row["correct"] for row in predictions) == metrics["correct"]
    tp = Counter(row["actual_index"] for row in predictions if row["correct"])
    actual = Counter(row["actual_index"] for row in predictions); predicted = Counter(row["predicted_index"] for row in predictions)
    independent_f1 = sum(2 * tp[i] / (actual[i] + predicted[i]) if actual[i] + predicted[i] else 0 for i in range(177)) / 177
    assert abs(independent_f1 - metrics["macro_f1"]) < 1e-12
    groups = defaultdict(list); class_groups = defaultdict(set)
    for row in predictions:
        groups[row["group_id"]].append(row)
        class_groups[row["actual_index"]].add(row["group_id"])
    assert all(len({row["actual_index"] for row in members}) == 1 for members in groups.values())
    metrics["known_group_count"] = len(groups)
    metrics["group_balanced_accuracy"] = sum(sum(row["correct"] for row in members) / len(members) for members in groups.values()) / len(groups)
    metrics["group_metric_definition"] = "Mean correct fraction per known/inferred related-image group; group identity does not prove biological independence"
    for row in metrics["per_class"]:
        row["group_support"] = len(class_groups[row["class_index"]])
    metrics["classes_with_fewer_than_10_test_images"] = sum(row["support"] < 10 for row in metrics["per_class"])
    atomic_json(directory / "test_metrics.json", metrics)
    rule = json.loads((ROOT / "data/manifests/leafcheck177_rule.json").read_text())
    baseline_matrix = np.zeros((177, 177), np.int64)
    baseline_matrix[:, rule["class_index"]] = np.bincount(test.labels, minlength=177)
    baseline = metrics_from_matrix(baseline_matrix, frozen["classes"])
    baseline.update({"rule": rule, "dataset_fingerprint": frozen["fingerprint"], "evaluation_split": "same frozen test images"})
    baseline["known_group_count"] = len(groups)
    baseline["group_balanced_accuracy"] = sum(members[0]["actual_index"] == rule["class_index"] for members in groups.values()) / len(groups)
    atomic_json(directory / "baseline_metrics.json", baseline)
    evaluation_figures(directory, metrics, predictions, frozen["classes"])
    learning_curves(directory)
    # Measure end-to-end CPU batch-one inference separately from training speed.
    samples = [ROOT / "data/processed/leafcheck177_final" / test.rows[i]["output_relative_path"] for i in [0, 1000, 4000, 8000]]
    source_cases = []
    for index, path in zip([0, 1000, 4000, 8000], samples):
        with Image.open(path) as image:
            source_cases.append({"image_id": test.rows[index]["image_id"], "image_path": test.rows[index]["output_relative_path"],
                                 "width": image.width, "height": image.height, "format": image.format,
                                 "frames": getattr(image, "n_frames", 1)})
    load_start = time.perf_counter(); measured = Predictor(directory, "cpu"); load_ms = (time.perf_counter() - load_start) * 1000
    for i in range(8): measured.predict(samples[i % len(samples)])
    timings = []
    for i in range(48):
        start = time.perf_counter(); measured.predict(samples[i % len(samples)]); timings.append((time.perf_counter() - start) * 1000)
    atomic_json(directory / "latency.json", {"device": "cpu", "hardware": hardware(), "torch_threads": torch.get_num_threads(),
                "batch_size": 1, "input_size": predictor.spec["input_size"], "warmup_calls": 8, "timed_calls": 48,
                "includes": "File decode, frame zero, EXIF, RGB, letterbox, scaling/normalization, network and softmax; local file cache warm",
                "p50_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
                "mean_ms": float(np.mean(timings)), "model_loading_ms_warm_filesystem": load_ms,
                "checkpoint_bytes": (directory / "model.pt").stat().st_size, "source_cases": source_cases, "measurements_ms": timings})
    artifact = json.loads((directory / "artifact.json").read_text())
    artifact["final_evaluation_complete"] = True
    artifact["files"] = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob("*"))
                         if p.is_file() and p.name != "artifact.json" and "__pycache__" not in p.parts}
    atomic_json(directory / "artifact.json", artifact)
    print(json.dumps({key: metrics[key] for key in ["correct", "sample_count", "accuracy", "macro_f1", "loss"]}, indent=2))
    print("Rule:", baseline["correct"], "/", baseline["sample_count"], baseline["accuracy"], baseline["macro_f1"])


if __name__ == "__main__": main()
