"""Compare a registered train-only feature augmentation on fixed development data."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now
from .data import PixelCache, release
from .field_expansion import load_inputs, prepare_development, validate
from .inference import Predictor
from .pretoria_baseline import PLAN as PRETORIA_PLAN, RAW as PRETORIA_RAW
from .train import device_for
from .upload_study import ARTIFACT, pixels, read, rows


PLAN = ROOT / "configs/leafcheck_feature_style_v1.json"
SOURCES = ("sheikh", "mduma", "pakistan", "pretoria")


def arm_config(plan, arm):
    return dict(read(ROOT / "configs" / plan["recipe"]),
                experiment=f"field-v11-{arm}", hypothesis=plan["hypothesis"],
                feature_style_probability=plan["arms"][arm], feature_style_alpha=plan["alpha"],
                min_epochs=plan["epochs"], max_epochs=plan["epochs"], seed=plan["seed"])


def development_rows(checkpoint):
    lineage = {"initial_weights": checkpoint}
    data = ROOT / "data/processed"
    _, field, old, new = load_inputs(data / "leafcheck_field_v2_sampled", lineage)
    _, pakistan_release, _, pak = load_inputs(data / "leafcheck_pakistan_tomato_dev_v1", lineage)
    pretoria_plan = read(PRETORIA_PLAN)
    manifest = ROOT / pretoria_plan["manifest"]
    if sha256(manifest) != pretoria_plan["manifest_sha256"]:
        raise ValueError("Pretoria development membership changed")
    pretoria = [dict(r, path=PRETORIA_RAW / f"{int(r['source_id'])}.jpg") for r in rows(manifest)
                if r["screen"] == "no_candidate"]
    if len(pretoria) != pretoria_plan["development_count"]:
        raise ValueError("Unexpected Pretoria development support")
    fingerprints = [field["fingerprint"], pakistan_release["fingerprint"], pretoria_plan["manifest_sha256"]]
    return (old, new, pak, pretoria), fingerprints


def development_pixels(members, size, class_mapping):
    old, new, pak, pretoria = members
    dev = prepare_development(old, new, size)
    dev["pakistan"] = (pixels(pak, size), np.array([int(r["class_id"]) for r in pak]))
    dev["pretoria"] = (pixels(pretoria, size), np.array([class_mapping[r["label"]] for r in pretoria]))
    return {key: dev[key] for key in SOURCES}


def compare(run, output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    device = device_for("auto")
    plan, frozen = read(PLAN), release()
    checkpoint = plan["reference_checkpoint"]
    if frozen["fingerprint"] != plan["dataset"] or sha256(ARTIFACT / "model.pt") != checkpoint:
        raise ValueError("The comparison's reference weights or dataset changed")
    paths = {"deployed": ARTIFACT}
    for name in plan["arms"]:
        source = run / name
        receipt = read(source / "status.json")
        if (receipt["state"] != "complete" or not receipt["checkpoint_reload_verified"]
                or receipt["completed_epochs"] != plan["epochs"]
                or read(source / "config.json") != arm_config(plan, name)
                or read(source / "dataset.json")["fingerprint"] != frozen["fingerprint"]):
            raise ValueError(f"The {name} run differs from the registered plan")
        paths[name] = source / "best_model.pt"
    cohort, fingerprints = development_rows(checkpoint)
    members, measurements = {}, {}
    for name, path in paths.items():
        predictor = Predictor(path, device)
        size = predictor.spec["input_size"]
        if size not in members:
            dev = development_pixels(cohort, size, frozen["class_to_idx"])
            members[size] = PixelCache("val", size, frozen), dev
        validation, dev = members[size]
        result = validate(predictor.model, validation, dev, predictor.classes, predictor.spec,
                          device, {"batch_size": 128}, "d4_probability_mean")
        result.update(checkpoint_sha256=sha256(predictor.path), epoch=predictor.metadata["selected_epoch"],
                      input_size=size)
        measurements[name] = result
        print(name, "master", result["master"]["macro_f1"], "field", result["selection_field_f1"], flush=True)
    before, after = (measurements[key] for key in ("control", "feature_style"))
    gain = after["selection_field_f1"] - before["selection_field_f1"]
    source_gains = {key: after["fields"][key]["macro_f1"] - before["fields"][key]["macro_f1"] for key in SOURCES}
    master_delta = {key: after["master"][key] - before["master"][key] for key in ("macro_f1", "accuracy")}
    rules = plan["advance"]
    passed = (gain >= rules["min_field_macro_f1_gain"]
              and min(source_gains.values()) >= -rules["max_field_source_macro_f1_drop"]
              and master_delta["macro_f1"] >= -rules["max_master_macro_f1_drop"]
              and master_delta["accuracy"] >= -rules["max_master_accuracy_drop"])
    atomic_json(output, {"plan_sha256": sha256(PLAN), "dataset_fingerprint": frozen["fingerprint"],
                         "field_fingerprints": fingerprints,
                         "measurements": measurements, "field_macro_f1_gain": gain, "source_gains": source_gains,
                         "master_deltas": master_delta, "passed_pilot": passed, "test_opened": False,
                         "device": str(device), "recorded_utc": utc_now()})
    print("Field gain", gain, "master changes", master_delta, "passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.run, args.output)
