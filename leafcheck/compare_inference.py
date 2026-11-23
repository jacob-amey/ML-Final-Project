"""Compare predeclared inference policies on validation for both incumbent seeds."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import DATA, ROOT, atomic_json, better, environment, sha256, utc_now, write_csv
from .data import PixelCache, release, split_rows
from .inference import Predictor
from .preprocessing import image_tensor
from .scoring import PredictionScores
from .train import device_for, evaluate


def summary(metrics):
    result = {key: metrics[key] for key in ("sample_count", "correct", "accuracy", "macro_f1", "loss")}
    result["zero_recall_classes"] = sum(row["recall"] == 0 for row in metrics["per_class"])
    rare = [row["f1"] for row in metrics["per_class"] if row["support"] < 20]
    result["under_20_support_mean_f1"] = float(np.mean(rare))
    return result


def check_latency(run, cache, policy):
    predictor = Predictor(run / "best_model.pt", "cpu")
    indexes = [0, len(cache) // 3, 2 * len(cache) // 3, len(cache) - 1]
    images = [DATA / cache.rows[i]["output_relative_path"] for i in indexes]
    measurements = []
    with torch.inference_mode():
        for choice in ({"name": "single_view"}, policy):
            scorer = PredictionScores(predictor.model, choice).eval()
            for image in images:
                scorer(image_tensor(image, predictor.spec)).softmax(1)
            times = []
            for image in images * 4:
                started = time.perf_counter()
                scorer(image_tensor(image, predictor.spec)).softmax(1)
                times.append(1000 * (time.perf_counter() - started))
            measurements.append({"name": choice["name"], "p50_ms": float(np.median(times)),
                                 "p95_ms": float(np.percentile(times, 95)), "calls": len(times)})
    return measurements


def compare(pair_file, output, device_name="auto"):
    if (ROOT / "artifacts/leafcheck177-v1/freeze.json").exists():
        raise RuntimeError("The final model is frozen. Validation selection is closed.")
    pair = json.loads(Path(pair_file).read_text())
    assert pair["evaluation_split"] == "validation only" and len(pair["runs"]) == 2
    frozen = release()
    expected = json.loads((ROOT / "data/manifests/leafcheck177_frozen.json").read_text())
    assert frozen["fingerprint"] == expected["fingerprint"]
    counts = np.bincount([int(row["class_id"]) for row in split_rows("train")], minlength=177).tolist()
    policies = [{"name": "single_view"}, {"name": "train_prior_quarter", "training_counts": counts},
                {"name": "d4_probability_mean"}]
    device = device_for(device_name)
    torch.set_num_threads(4)
    cache = PixelCache("val", pair["input_size"], frozen)
    assert len(cache) == 16836
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {policy["name"]: [] for policy in policies}
    for record in pair["runs"]:
        run = ROOT / "runs/final177" / record["run"]
        checkpoint = run / "best_model.pt"
        status = json.loads((run / "status.json").read_text())
        assert status["state"] == "complete" and status["checkpoint_reload_verified"]
        assert sha256(checkpoint) == record["checkpoint_sha256"]
        predictor = Predictor(checkpoint, device)
        assert predictor.metadata["dataset_fingerprint"] == frozen["fingerprint"]
        assert predictor.spec["input_size"] == pair["input_size"]
        for policy in policies:
            name = policy["name"]
            started = time.perf_counter()
            model = PredictionScores(predictor.model, policy).to(device).eval()
            metrics, rows = evaluate(model, cache, predictor.spec, device, frozen["classes"], 256, True)
            result = {**summary(metrics), "run": run.name, "seed": record["seed"],
                      "epoch": status["best"]["epoch"], "checkpoint_sha256": record["checkpoint_sha256"],
                      "seconds": time.perf_counter() - started}
            if name == "single_view":
                if device.type == "mps":
                    assert abs(result["macro_f1"] - record["macro_f1"]) < 1e-12
                else:
                    assert abs(result["macro_f1"] - record["macro_f1"]) < .001
            else:
                control = results["single_view"][-1]
                result["macro_f1_gain"] = result["macro_f1"] - control["macro_f1"]
                result["changed_predictions"] = sum(row["predicted_index"] != before["predicted_index"]
                                                    for row, before in zip(rows, control_rows))
                result["corrected_predictions"] = sum(row["correct"] and not before["correct"] for row, before in zip(rows, control_rows))
                result["new_mistakes"] = sum(before["correct"] and not row["correct"] for row, before in zip(rows, control_rows))
                original = json.loads((output / f"{run.name}-single_view-metrics.json").read_text())
                result["per_class_f1_gains"] = {row["class_name"]: row["f1"] - before["f1"]
                                               for row, before in zip(metrics["per_class"], original["per_class"])}
            results[name].append(result)
            if name == "single_view":
                control_rows = rows
            atomic_json(output / f"{run.name}-{name}-metrics.json", metrics)
            write_csv(output / f"{run.name}-{name}-predictions.csv", rows)
            print(json.dumps({"policy": name, **{k:v for k,v in result.items() if k != "per_class_f1_gains"}}), flush=True)
    decisions = []
    winner = policies[0]
    eligible = []
    for policy in policies[1:]:
        name = policy["name"]
        gains = [r["macro_f1_gain"] for r in results[name]]
        mean_gain = sum(gains) / len(gains)
        passes = mean_gain >= .005 and min(gains) > 0
        timing = None
        if passes and name == "d4_probability_mean":
            timing = check_latency(ROOT / "runs/final177" / results[name][0]["run"], cache, policy)
            passes = timing[-1]["p95_ms"] <= 500
        if passes:
            eligible.append((mean_gain, policy))
        decisions.append({"policy": name, "paired_gains": gains, "mean_gain": mean_gain,
                          "accepted_by_rule": passes, "cpu_single_photo_latency": timing})
    if eligible:
        winner = max(eligible, key=lambda candidate: candidate[0])[1]
    winning_result = None
    for candidate in sorted(results[winner["name"]], key=lambda row: row["epoch"]):
        if better(candidate, winning_result):
            winning_result = candidate
    decision = {"recorded_utc": utc_now(), "evaluation_split": "validation only", "incumbent_pair": str(pair_file),
                "dataset_fingerprint": frozen["fingerprint"], "environment": environment(device),
                "evidence_directory": str(output), "test_predictions_used": False, "policies": decisions,
                "results": results, "selected_policy": winner,
                "selected_run": winning_result["run"], "selected_checkpoint_sha256": winning_result["checkpoint_sha256"],
                "deployment_validation": {key: winning_result[key] for key in ("macro_f1", "accuracy", "loss", "sample_count", "correct")}}
    atomic_json(ROOT / "docs/inference_policy_comparison.json", decision)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = parser.parse_args()
    compare(args.pair, args.output, args.device)
