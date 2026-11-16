"""Prepare a separately versioned longer-budget experiment without overwriting its parent."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import torch
from .common import ROOT, atomic_checkpoint, atomic_json, environment, sha256, utc_now


def prepare(parent, output, max_epochs):
    parent = Path(parent); output = Path(output)
    status = json.loads((parent / "status.json").read_text())
    assert status["state"] == "complete" and status["checkpoint_reload_verified"]
    original_config = json.loads((parent / "config.json").read_text())
    assert max_epochs > original_config["max_epochs"]
    last = torch.load(parent / "last.pt", map_location="cpu", weights_only=True)
    assert last["epoch"] == status["completed_epochs"] + 1 and last["next_batch"] == 0
    assert last["config"] == original_config
    config = copy.deepcopy(original_config)
    config.update({"experiment": original_config["experiment"] + "-longer-budget",
                   "hypothesis": "The leading 48-epoch recipe was still improving late; continue its exact optimizer/RNG state to test whether a longer budget yields meaningful validation gains.",
                   "reference": parent.name + "; only the maximum epoch budget changes, with the same validation-led early stop and scheduler",
                   "max_epochs": max_epochs})
    best = torch.load(parent / "best_model.pt", map_location="cpu", weights_only=True)
    last["config"] = config
    # The parent's last update checkpoint precedes its final reload verification.
    # Include that completed procedure cost when reporting cumulative runtime.
    last["training_seconds"] = status["training_seconds"]
    best["config"] = config  # Procedure metadata changes; the parent tensors remain untouched.
    output.mkdir(parents=True, exist_ok=False)
    for name in ["dataset.json", "preprocessing.json", "class_to_idx.json", "class_weights.json", "history.csv", "validation_metrics.json", "validation_predictions.csv"]:
        shutil.copyfile(parent / name, output / name)
    atomic_json(output / "config.json", config)
    parent_environment = json.loads((parent / "environment.json").read_text())
    atomic_json(output / "environment.json", environment(parent_environment["device"]))
    shutil.copytree(ROOT / "leafcheck", output / "source/leafcheck", ignore=shutil.ignore_patterns("__pycache__"))
    atomic_checkpoint(output / "last.pt", last)
    atomic_checkpoint(output / "best_model.pt", best)
    lineage = {"kind":"longer_budget", "created_utc":utc_now(), "parent_run":parent.name,
               "parent_status":status,"parent_environment":parent_environment,"parent_config":original_config,
               "parent_best_sha256":sha256(parent / "best_model.pt"),"parent_last_sha256":sha256(parent / "last.pt"),
               "parent_training_seconds":status["training_seconds"],"continuation_epoch":last["epoch"],
               "changed_training_settings":["max_epochs"],"all_parent_model_optimizer_scheduler_rng_tensors_preserved":True}
    lineage["timing_metadata_includes_parent_completion_verification"] = True
    atomic_json(output / "lineage.json", lineage)
    atomic_json(output / "status.json", {"state":"ready_to_resume", "parent_run":parent.name})
    print(f"Prepared {output}. Resume it with python -m leafcheck.train --run {output} --resume", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-epochs",type=int,required=True)
    args=parser.parse_args();prepare(args.parent,args.output,args.max_epochs)
