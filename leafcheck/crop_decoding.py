"""Compare fixed crop-first decoding policies using established development splits."""

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now
from .data import PixelCache
from .field_expansion import load_inputs, prepare_development
from .inference import Predictor
from .potato_policy import distribution, master_probabilities
from .train import device_for
from .upload_study import ARTIFACT, read, score, summarize


PLAN = ROOT / "configs/leafcheck_crop_decoding_v1.json"


def decode(probabilities, crop_index):
    membership = np.eye(int(crop_index.max()) + 1, dtype=probabilities.dtype)[crop_index]
    crop_scores = probabilities @ membership
    return {
        "joint": probabilities.argmax(1),
        "crop_reward": (probabilities * np.sqrt(crop_scores[:, crop_index])).argmax(1),
        "crop_first": np.where(
            crop_scores.argmax(1)[:, None] == crop_index, probabilities, -1
        ).argmax(1),
    }


def compare(probabilities, labels, classes, crop_index):
    report = {}
    for method, predicted in decode(probabilities, crop_index).items():
        result = score(labels, predicted, classes, np.unique(labels))
        summary = summarize(result)
        summary["crop_accuracy"] = 1 - result["wrong_crop"] / len(labels)
        summary["balanced_accuracy"] = float(np.mean([result["per_class"][i]["recall"] for i in set(labels)]))
        summary["per_class"] = [row for row in result["per_class"] if row["support"]]
        summary["blight"] = [row for row in result["blight"] if row["total"]]
        summary["largest_errors"] = result["largest_errors"]
        report[method] = summary
    return report


def run(output, device):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    started = time.monotonic()
    plan = read(PLAN)
    frozen, release, previous, new = load_inputs(
        ROOT / plan["field_release"], {"initial_weights": plan["checkpoint_sha256"]}
    )
    if release["fingerprint"] != plan["field_fingerprint"]:
        raise ValueError("Unexpected development release")
    predictor = Predictor(ARTIFACT, device)
    classes, spec = predictor.classes, predictor.spec
    _, crop_index = np.unique([label.split("__", 1)[0] for label in classes], return_inverse=True)
    cache = PixelCache("val", spec["input_size"], frozen)
    master = compare(master_probabilities(predictor, cache), cache.labels, classes, crop_index)
    field = {}
    for source, (images, labels) in prepare_development(previous, new, spec["input_size"]).items():
        field[source] = compare(
            distribution(predictor.scoring_model, images, spec, device), labels, classes, crop_index
        )
    baseline = master["joint"]
    decisions = {}
    for method in ("crop_reward", "crop_first"):
        sources = [field[name] for name in plan["field_selection_sources"]]
        gains = [r[method]["macro_f1"] - r["joint"]["macro_f1"] for r in sources]
        gain = float(np.mean(gains))
        crop_gain = float(np.mean([r[method]["crop_accuracy"] - r["joint"]["crop_accuracy"] for r in sources]))
        drift = {key: master[method][key] - baseline[key] for key in ("macro_f1", "accuracy")}
        decisions[method] = {
            "field_f1_gain": gain, "per_source_f1_gains": gains, "field_crop_gain": crop_gain,
            "master_delta": drift, "passes": (
                gain >= plan["min_field_f1_gain"] and min(gains) >= -plan["max_field_source_drop"]
                and crop_gain >= plan["min_field_crop_accuracy_gain"]
                and min(drift.values()) >= -plan["max_master_drop"]
            ),
        }
    result = {
        "checkpoint": plan["checkpoint_sha256"], "plan_sha256": sha256(PLAN),
        "dataset": frozen["fingerprint"], "field_release": release["fingerprint"],
        "evaluated_utc": utc_now(), "device": str(device), "elapsed_seconds": time.monotonic() - started,
        "master": master, "field": field, "decisions": decisions, "test_opened": False,
    }
    atomic_json(output, result)
    print(decisions, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.output, device_for(args.device))
