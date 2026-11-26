"""Verify and compare the registered scratch rice pilots on development data."""

import argparse
from collections import Counter, defaultdict
import math
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, metrics_from_matrix, sha256, utc_now, write_csv
from .data import PixelCache, release
from .inference import Predictor
from .scoring import PredictionScores
from .train import device_for, evaluate
from .upload_study import ARTIFACT, read


PLAN = ROOT / "configs/leafcheck_rice_field_study_v1.json"
ARMS = ("matched_repeat", "new_field")
POLICY = {"name": "d4_probability_mean"}


def verify_inputs(plan, frozen):
    registries = {}
    for key in ("training", "validation"):
        path = ROOT / plan[key]["path"]
        if sha256(path) != plan[key]["sha256"]:
            raise ValueError(f"Changed registered {key} membership")
        record = read(path)
        if record["base_fingerprint"] != frozen["fingerprint"]:
            raise ValueError(f"Changed base dataset for {key}")
        registries[key] = record
    training, validation = registries["training"], registries["validation"]
    rows = validation["records"]
    ids = {row["sha256"] for row in rows}
    if (len(ids) != len(rows) or ids.intersection(row["sha256"] for row in training["members"])
            or set(row["source"] for row in rows) != {"hasan", "banglarice"}
            or validation["training_sha256"] != plan["training"]["sha256"]):
        raise ValueError("Validation membership overlaps or disagrees with the study")
    for row in rows:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError(f"Validation photo changed: {row['member']}")
    return training, rows


def receipt(run, arm, config, dataset, count, plan):
    folder = run / arm
    actual = read(folder / "config.json")
    state = read(folder / "status.json")
    if (config != actual or state["state"] != "complete" or not state["checkpoint_reload_verified"]
            or state["completed_epochs"] != plan["epochs"]
            or state["total_steps"] != plan["epochs"] * math.ceil(count / config["batch_size"])
            or read(folder / "dataset.json")["fingerprint"] != dataset):
        raise ValueError(f"Incomplete or mismatched {arm} run")
    return state


def field_score(predictor, rows):
    by_source = defaultdict(list)
    for row in rows:
        output = predictor.predict(ROOT / row["path"])
        predicted = output["label"]
        by_source[row["source"]].append(dict(
            row, predicted=predicted, score=output["score"], correct=predicted == row["class_name"],
            wrong_crop=not predicted.startswith("rice__")))
    result = {}
    for source, members in by_source.items():
        positions = {cls: i for i, cls in enumerate(predictor.classes)}
        matrix = np.zeros((len(positions), len(positions)), dtype=np.int64)
        for row in members:
            matrix[positions[row["class_name"]], positions[row["predicted"]]] += 1
        metrics = metrics_from_matrix(matrix, predictor.classes)
        represented = [v for v in metrics["per_class"] if v["support"]]
        group_support = Counter(row["group"] for row in members)
        group_correct = Counter(row["group"] for row in members if row["correct"])
        result[source] = {
            "correct": metrics["correct"], "support": metrics["sample_count"], "accuracy": metrics["accuracy"],
            "macro_f1": float(np.mean([v["f1"] for v in represented])), "represented_classes": represented,
            "wrong_crop": sum(row["wrong_crop"] for row in members),
            "equal_group_accuracy": sum(group_correct[group] / n for group, n in group_support.items()) / len(group_support),
            "predicted_crops": Counter(row["predicted"].split("__")[0] for row in members),
            "capture_group_support": group_support,
        }
    return result, [row for members in by_source.values() for row in members]


def compare(run, output):
    output.mkdir(parents=True, exist_ok=True)
    target = output / "result.json"
    if target.exists():
        raise FileExistsError(target)
    torch.set_num_threads(4)
    device = device_for("auto")
    frozen, plan = release(), read(PLAN)
    training, rows = verify_inputs(plan, frozen)
    if sha256(ARTIFACT / "model.pt") != plan["reference_sha256"]:
        raise ValueError("The current deployment differs from the protocol")
    cached = PixelCache("val", plan["input_size"], frozen)
    configs = {name: read(ROOT / plan["configs"][name]) for name in ARMS}
    shared = [{k: v for k, v in configs[name].items() if k not in {"experiment", "training_addition"}} for name in ARMS]
    if shared[0] != shared[1]:
        raise ValueError("Arm configurations mismatch outside the registered intervention")
    receipts = {}
    paths = {"deployed": ARTIFACT}
    for arm in ARMS:
        options = configs[arm]["training_addition"]
        if (options["arm"] != arm or options["sha256"] != plan["training"]["sha256"]
                or options["manifest"] != plan["training"]["path"]):
            raise ValueError("Registered training addition mismatch")
        receipts[arm] = receipt(run, arm, configs[arm], frozen["fingerprint"], 59008 + len(training["members"]), plan)
        paths[arm] = run / arm / "best_model.pt"

    results = {}
    for name, path in paths.items():
        predictor = Predictor(path, device)
        predictor.scoring_model = PredictionScores(predictor.model, POLICY).to(device).eval()
        recorded = None
        if name in receipts:
            if predictor.metadata["selected_epoch"] != receipts[name]["best"]["epoch"]:
                raise ValueError(f"Selected checkpoint mismatch for {name}")
            validation, _ = evaluate(predictor.scoring_model, cached, predictor.spec, device, predictor.classes, 128)
            recorded = {k: validation[k] for k in ("accuracy", "macro_f1", "loss", "correct", "sample_count")}
            recorded["rice_metrics"] = [m for m in validation["per_class"] if m["class_name"].startswith("rice__")]
        fields, predicted = field_score(predictor, rows)
        csv_path = output / f"{name}-field-predictions.csv"
        write_csv(csv_path, predicted)
        results[name] = {"checkpoint_sha256": sha256(predictor.path), "epoch": predictor.metadata["selected_epoch"],
                         "resolution": predictor.spec["input_size"], "validation": recorded, "fields": fields,
                         "mean_field_macro_f1": float(np.mean([v["macro_f1"] for v in fields.values()])),
                         "prediction_file_sha256": sha256(csv_path), "run_state": receipts.get(name)}
        print(name, "field correct", {k: (v["correct"], v["support"]) for k, v in fields.items()},
              "master F1", recorded and recorded["macro_f1"], flush=True)

    control, candidate = (results[name] for name in ARMS)
    gains = {key: candidate["validation"][key] - control["validation"][key] for key in ("macro_f1", "accuracy")}
    gains["mean_field"] = candidate["mean_field_macro_f1"] - control["mean_field_macro_f1"]
    gains["mean_field_vs_deployed"] = candidate["mean_field_macro_f1"] - results["deployed"]["mean_field_macro_f1"]
    gains["per_source"] = {key: value["macro_f1"] - control["fields"][key]["macro_f1"]
                           for key, value in candidate["fields"].items()}
    guards = plan["advance"]
    advance = (
        gains["macro_f1"] >= guards["min_master_macro_f1_delta"]
        and gains["accuracy"] >= guards["min_master_accuracy_delta"]
        and gains["mean_field"] >= guards["min_mean_field_gain"]
        and gains["mean_field_vs_deployed"] >= guards["min_gain_vs_deployed"]
        and min(gains["per_source"].values()) >= guards["min_per_source_gain"]
    )
    atomic_json(target, {"recorded_utc": utc_now(), "device": str(device), "plan_sha256": sha256(PLAN),
                         "master_fingerprint": frozen["fingerprint"], "training": plan["training"],
                         "validation": plan["validation"], "results": results, "deltas": gains,
                         "advance": advance, "course_test_opened": False, "deployment_changed": False})
    print("deltas", gains, "advance", advance, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.run, args.output)
