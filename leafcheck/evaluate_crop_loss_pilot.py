"""Compare the matched crop-loss arms on development data only."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now
from .data import PixelCache, release
from .field_expansion import load_inputs, prepare_development, validate
from .inference import Predictor
from .train import device_for
from .upload_study import ARTIFACT, pixels, read


PLAN = ROOT / "configs/leafcheck_crop_loss_pilot_v1.json"
SOURCES = ("sheikh", "mduma", "pakistan")
DATA = ROOT / "data/processed"


def verify_arms(plan, run, frozen):
    reference_config = read(ROOT / "configs" / plan["recipe"])
    for name, weight in plan["arms"].items():
        directory = run / name
        status = read(directory / "status.json")
        config = read(directory / "config.json")
        expected = {
            **reference_config, "experiment": "field-v8-crop-loss-" + name,
            "hypothesis": plan["hypothesis"], "crop_loss_weight": weight,
            "max_epochs": plan["epochs"], "min_epochs": plan["epochs"], "seed": plan["seed"],
        }
        if (status["state"] != "complete" or status["completed_epochs"] != plan["epochs"]
                or not status["checkpoint_reload_verified"] or config != expected
                or read(directory / "dataset.json")["fingerprint"] != frozen["fingerprint"]):
            raise ValueError(f"The completed {name} run differs from the frozen comparison")


def compare(run, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    plan, frozen = read(PLAN), release()
    if frozen["fingerprint"] != plan["dataset"] or sha256(ARTIFACT / "model.pt") != plan["reference_checkpoint"]:
        raise ValueError("Reference checkpoint or dataset differs from the plan")
    verify_arms(plan, run, frozen)
    lineage = {"initial_weights": plan["reference_checkpoint"]}
    _, field, old, new = load_inputs(DATA / "leafcheck_field_v2_sampled", lineage)
    _, pakistan, _, newer = load_inputs(DATA / "leafcheck_pakistan_tomato_dev_v1", lineage)
    device = device_for("auto")
    members = {}
    measurements = {}
    paths = {"deployed": ARTIFACT, **{name: run / name / "best_model.pt" for name in plan["arms"]}}
    for name, path in paths.items():
        predictor = Predictor(path, device)
        size = predictor.spec["input_size"]
        if size not in members:
            dev = prepare_development(old, new, size)
            dev["pakistan"] = (pixels(newer, size), np.array([int(row["class_id"]) for row in newer]))
            members[size] = PixelCache("val", size, frozen), dev
        validation, dev = members[size]
        result = validate(predictor.model, validation, dev, predictor.classes, predictor.spec, device,
                          {"batch_size": 128, "tracked_labels": ["tomato__early_blight", "tomato__late_blight",
                          "tomato__healthy", "potato__early_blight", "potato__late_blight", "potato__healthy"]},
                          "d4_probability_mean")
        result["checkpoint"] = str(predictor.path)
        result["sha256"] = sha256(predictor.path)
        result["epoch"] = predictor.metadata["selected_epoch"]
        measurements[name] = result
        print(name, "master", result["master"]["macro_f1"], "field", result["selection_field_f1"], flush=True)
    control, candidate = (measurements[name] for name in ("control", "crop_loss"))
    gain = candidate["selection_field_f1"] - control["selection_field_f1"]
    source_gains = {name: candidate["fields"][name]["macro_f1"] - control["fields"][name]["macro_f1"]
                    for name in SOURCES}
    crop_gain = float(np.mean([
        (control["fields"][name]["wrong_crop"] - candidate["fields"][name]["wrong_crop"]) /
        control["fields"][name]["sample_count"] for name in SOURCES
    ]))
    master = {metric: candidate["master"][metric] - control["master"][metric] for metric in ("macro_f1", "accuracy")}
    passed = (gain >= plan["min_field_gain_vs_control"] and
              master["macro_f1"] >= -plan["max_master_macro_f1_drop_vs_control"] and
              master["accuracy"] >= -plan["max_master_accuracy_drop_vs_control"] and
              min(source_gains.values()) >= -plan["max_field_source_macro_f1_drop_vs_control"] and
              crop_gain >= plan["min_field_wrong_crop_rate_reduction"])
    atomic_json(output, {
        "plan_sha256": sha256(PLAN), "dataset_fingerprint": frozen["fingerprint"],
        "field_fingerprints": [field["fingerprint"], pakistan["fingerprint"]],
        "verified_utc": utc_now(), "device": str(device), "test_opened": False,
        "measurements": measurements, "field_joint_f1_gain": gain, "field_source_gains": source_gains,
        "wrong_crop_rate_reduction": crop_gain, "master_deltas": master, "passed_pilot": passed,
    })
    print("Field gain", gain, "crop error reduction", crop_gain, "master", master, "passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.run, args.output)
