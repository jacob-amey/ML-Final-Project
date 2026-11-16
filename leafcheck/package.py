"""Freeze an explicitly selected completed run before its one final evaluation."""

import argparse
import json
from pathlib import Path
import shutil
import torch
from .common import ROOT, atomic_json, sha256, utc_now
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
    for name in ["config.json", "environment.json", "class_to_idx.json", "preprocessing.json", "validation_metrics.json", "history.csv", "status.json"]:
        shutil.copyfile(run / name, destination / name)
    runtime = destination / "runtime/leafcheck"; runtime.mkdir(parents=True)
    for name in ["__init__.py", "model.py", "preprocessing.py", "inference.py"]:
        shutil.copyfile(ROOT / "leafcheck" / name, runtime / name)
    freeze = {"model_version": "leafcheck177-v1", "created_utc": utc_now(), "run": str(run.resolve()),
              "checkpoint_sha256": sha256(destination / "model.pt"), "dataset_fingerprint": frozen["fingerprint"],
              "config_sha256": sha256(destination / "config.json"), "preprocessing_sha256": sha256(destination / "preprocessing.json"),
              "class_mapping_sha256": sha256(destination / "class_to_idx.json"), "selected_epoch": checkpoint["epoch"],
              "selection": checkpoint["selection"], "validation_macro_f1": checkpoint["validation"]["macro_f1"],
              "rationale": rationale, "confidence_policy": "Display softmax scores; no calibrated confidence or unknown-input threshold",
              "evaluation_device": "cpu", "evaluation_batch_size": 128, "test_predictions_used_in_selection": False}
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
