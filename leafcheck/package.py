"""Freeze an explicitly selected completed run before its one final evaluation."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import torch
from .common import ROOT, DATA, atomic_json, sha256, utc_now
from .data import release


def package(run, destination, rationale):
    run = Path(run); destination = Path(destination)
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "complete" and status["checkpoint_reload_verified"]
    checkpoint = torch.load(run / "best_model.pt", weights_only=True, map_location="cpu")
    frozen = release()
    assert checkpoint["dataset_fingerprint"] == frozen["fingerprint"]
    if destination.exists():
        raise FileExistsError("A frozen artifact already exists. Preserve it; use a new version before any new test protocol.")
    destination.mkdir(parents=True)
    shutil.copyfile(run / "best_model.pt", destination / "model.pt")
    for name in ["config.json", "environment.json", "dataset.json", "class_to_idx.json", "class_weights.json", "preprocessing.json", "validation_metrics.json", "history.csv", "status.json"]:
        shutil.copyfile(run / name, destination / name)
    if (run / "lineage.json").exists():
        shutil.copyfile(run / "lineage.json", destination / "lineage.json")
    shutil.copyfile(DATA / "dataset_info.json", destination / "dataset_release_info.json")
    if (ROOT / "requirements-lock.txt").exists():
        shutil.copyfile(ROOT / "requirements-lock.txt", destination / "requirements-training-lock.txt")
    env = json.loads((run / "environment.json").read_text())
    (destination / "requirements-runtime.txt").write_text(f"torch=={env['torch']}\nPillow=={env['pillow']}\n")
    (destination / "README.md").write_text(
        "# Frozen LeafCheck model bundle\n\n"
        "The checkpoint, configuration, preprocessing and class order are paired. "
        "Use the included runtime/leafcheck package for inference; no notebooks or dataset are needed.\n\n"
        "Install requirements-runtime.txt in a compatible Python environment. "
        "Add this bundle's runtime directory to PYTHONPATH and run "
        "python -m leafcheck.inference --artifact /absolute/bundle/path /absolute/photo.jpg.\n\n"
        "The app installer validates artifact.json and creates an explicit model reference. "
        "freeze.json records selection before evaluation. Saved scores apply only to their declared datasets. "
        "Softmax scores are not calibrated correctness probabilities or definitive diagnoses.\n")
    external_protocol = ROOT / "data/manifests/leafcheck177_external_frozen.json"
    if external_protocol.exists():
        shutil.copyfile(external_protocol, destination / "external_protocol.json")
        shutil.copyfile(ROOT / "data/manifests/leafcheck177_external.csv", destination / "external_manifest.csv")
    runtime = destination / "runtime/leafcheck"; runtime.mkdir(parents=True)
    for name in ["__init__.py", "model.py", "preprocessing.py", "inference.py"]:
        shutil.copyfile(ROOT / "leafcheck" / name, runtime / name)
    # A training photo is a software-format fixture, never evaluation evidence.
    with (DATA / "manifest.csv").open() as file:
        candidates = [row for row in csv.DictReader(file) if row["status"] == "included" and row["split"] == "train"
                      and row["format"] == "MPO" and int(row["width"]) * int(row["height"]) <= 24000000
                      and (DATA / row["output_relative_path"]).stat().st_size < 10 * 1024 * 1024]
    if candidates:
        row = min(candidates, key=lambda row:(DATA / row["output_relative_path"]).stat().st_size)
        samples = destination / "samples"; samples.mkdir(exist_ok=True)
        target = samples / "format_mpo.jpg"; shutil.copyfile(DATA / row["output_relative_path"], target)
        assert sha256(target) == row["sha256"]
        atomic_json(destination / "integration_fixtures.json", [{"sample_file": "samples/format_mpo.jpg",
                    "source_image_id": row["image_id"], "source_split": "train", "frames": int(row["frames"]),
                    "purpose": "First-frame MPO/JPEG software compatibility only; not a scored evaluation example."}])
    freeze = {"model_version": "leafcheck177-v1", "created_utc": utc_now(), "run": str(run.resolve()),
              "checkpoint_sha256": sha256(destination / "model.pt"), "dataset_fingerprint": frozen["fingerprint"],
              "config_sha256": sha256(destination / "config.json"), "preprocessing_sha256": sha256(destination / "preprocessing.json"),
              "class_mapping_sha256": sha256(destination / "class_to_idx.json"), "selected_epoch": checkpoint["epoch"],
              "selection": checkpoint["selection"], "validation_macro_f1": checkpoint["validation"]["macro_f1"],
              "rationale": rationale, "confidence_policy": "Display softmax scores; no calibrated confidence or unknown-input threshold",
              "evaluation_device": "cpu", "evaluation_batch_size": 128, "test_predictions_used_in_selection": False}
    if external_protocol.exists():
        freeze["external_protocol_sha256"] = sha256(destination / "external_protocol.json")
        freeze["external_manifest_sha256"] = sha256(destination / "external_manifest.csv")
    atomic_json(destination / "freeze.json", freeze)
    atomic_json(destination / "artifact.json", {**freeze, "files": {
        str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()}})
    print(json.dumps(freeze, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rationale", required=True)
    args = parser.parse_args(); package(args.run, args.output, args.rationale)


if __name__ == "__main__": main()
