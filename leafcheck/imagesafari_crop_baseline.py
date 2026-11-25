"""Measure crop recognition on the site-held Image Safari development groups."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, metrics_from_matrix, sha256, utc_now, write_csv
from .crop_head_study import blend_crop
from .inference import Predictor
from .potato_policy import distribution
from .potato_study import verify_arm
from .train import device_for
from .upload_study import ARTIFACT, network, pixels, read, rows


NAME = "leafcheck_imagesafari_crop_v1"
DATA = ROOT / "data/processed" / NAME
RECEIPT = ROOT / f"data/manifests/{NAME}_release.json"


def crop_metrics(truth, predicted, masked, crops):
    n = len(crops)
    matrix = np.bincount(truth[masked] * n + predicted[masked], minlength=n * n).reshape(n, n)
    report = metrics_from_matrix(matrix, crops)
    report["represented_macro_f1"] = float(np.mean([row["f1"] for row in report["per_class"] if row["support"]]))
    report["per_class"] = [row for row in report["per_class"] if row["support"]]
    return report


def run(output, selection):
    if (output / "metrics.json").exists() or (output / "predictions.csv").exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    device = device_for("auto")
    seal = read(RECEIPT)
    actual = {name: sha256(DATA / name) for name in seal["files_sha256"]}
    if (read(DATA / "release.json") != seal or actual != seal["files_sha256"]
            or hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() != seal["fingerprint"]):
        raise ValueError("Image Safari release identity changed")
    members = [r for r in rows(DATA / "manifest.csv") if r["status"] == "included"]
    if ({r["group_id"] for r in members if r["split"] == "train"}
            & {r["group_id"] for r in members if r["split"] == "val"}):
        raise ValueError("A capture site crosses development splits")
    development = [dict(r, path=DATA / r["output_relative_path"]) for r in members if r["split"] == "val"]
    predictor = Predictor(ARTIFACT, device)
    receipt = read(selection)
    config = receipt["run"]["config"]
    arm = receipt["arms"]["field"]
    checkpoint = Path(arm["checkpoint"])
    if sha256(checkpoint) != arm["sha256"] or not arm["checkpoint_reload_verified"]:
        raise ValueError("Invalid crop checkpoint")
    if predictor.metadata["checkpoint_sha256"] != config["reference_checkpoint"]:
        raise ValueError("Base checkpoint changed")
    head = network(config, device)
    head.load_state_dict(verify_arm(checkpoint, config, receipt["run"])["weights"], strict=True)
    crops = config["labels"]
    indices = np.asarray([crops.index(label.split("__", 1)[0]) for label in predictor.classes])
    reference = distribution(predictor.scoring_model, pixels(development, predictor.spec["input_size"]),
                             predictor.spec, device)
    spec = receipt["run"]["crop_spec"]
    crop_probability = distribution(head, pixels(development, spec["input_size"]), spec, device)
    membership = np.eye(len(crops), dtype=reference.dtype)[indices]
    blend = blend_crop(reference, crop_probability, indices, config["crop_blend"])
    choices = {
        "automatic_model": indices[reference.argmax(1)],
        "original_crop_marginal": (reference @ membership).argmax(1),
        "crop_head_only": crop_probability.argmax(1),
        "crop_head_blend": indices[blend.argmax(1)],
    }
    truth = np.asarray([crops.index(r["crop"]) for r in development])
    summary = {key: crop_metrics(truth, value, slice(None), crops) for key, value in choices.items()}
    sites = np.array([r["site"] for r in development])
    per_site = {
        str(site): {key: crop_metrics(truth, value, sites == site, crops) for key, value in choices.items()}
        for site in np.unique(sites)
    }
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.csv"
    write_csv(prediction_path, [
        {**{k: r[k] for k in ("image_id", "site", "day", "crop", "group_id")},
         **{key: crops[value[i]] for key, value in choices.items()}}
        for i, r in enumerate(development)
    ])
    atomic_json(output / "metrics.json", {
        "completed": utc_now(), "device": str(device), "role": "validation_crop_only",
        "dataset": seal["fingerprint"], "reference": config["reference_checkpoint"], "crop_head": arm["sha256"],
        "selecting_run_sha256": sha256(selection), "predictions_sha256": sha256(prediction_path),
        "metrics": summary, "per_site": per_site, "test_opened": False,
    })
    print({key: (value["correct"], value["sample_count"], value["represented_macro_f1"]) for key, value in summary.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.selection)
