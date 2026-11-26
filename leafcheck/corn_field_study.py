"""Compare label-matched master replay with a new isolated corn training source."""

import argparse
import fcntl
from pathlib import Path
import signal

import numpy as np
import torch

from .common import ROOT, atomic_json, environment, sha256, utc_now
from .data import cache_path
from .field_expansion import load_inputs
from .inference import Predictor
from .potato_study import matched_training_rows, run_arm, score_pixels
from .pretoria_baseline import PLAN as DEVELOPMENT_PLAN, RAW as DEVELOPMENT_RAW
from .train import device_for
from .upload_study import ARTIFACT, load_study, pixels, read, rows


PLAN = ROOT / "configs/leafcheck_corn_field_v1.json"
ADDITION = ROOT / "data/processed/leafcheck_bangladesh_corn_v1"


def with_supported_average(result):
    result["macro_f1"] = float(np.mean([row["f1"] for row in result["per_class"] if row["support"]]))
    return result


def train(output, resume):
    if (output / "decision.json").exists():
        print("Preserving the completed run", flush=True)
        return
    config, field_plan = read(PLAN), read(DEVELOPMENT_PLAN)
    frozen, release, _, addition = load_inputs(ADDITION, {"initial_weights": field_plan["checkpoint_sha256"]})
    _, master, _ = load_study(read(ROOT / "configs/upload_blight_v1.json"))
    if any(r["split"] != "train" for r in addition):
        raise ValueError("Expected a training-only source")
    minimum = config["selection"]["minimum_source_per_class"]
    if len(release["train_only_counts"]) != 3 or min(release["train_only_counts"].values()) < minimum:
        raise ValueError("Insufficient training support after source screening")
    manifest = ROOT / field_plan["manifest"]
    if sha256(manifest) != field_plan["manifest_sha256"]:
        raise ValueError("Expected fixed Pretoria development membership")
    corn = config["labels"]
    included = [r for r in rows(manifest) if r["screen"] == "no_candidate"]
    if len(included) != field_plan["development_count"]:
        raise ValueError("Unexpected development source size")
    new_dev = [dict(r, path=DEVELOPMENT_RAW / f"{int(r['source_id'])}.jpg", class_name=r["label"])
               for r in included]
    base = [r for r in master if r["split"] == "train" and r["class_name"] in corn]
    development = {
        "master": [r for r in master if r["split"] == "val" and r["class_name"] in corn],
        "field": new_dev,
    }
    all_training = {r["sha256"] for r in base + addition}
    if len(all_training) != len(base) + len(addition) or any(
            all_training.intersection(r["sha256"] for r in subset) for subset in development.values()):
        raise ValueError("Overlap among training and development rows")
    device = device_for("auto")
    torch.set_num_threads(4)
    spec = read(cache_path(config["input_size"], frozen["fingerprint"]) / "preprocessing.json")
    run = {"recipe_sha256": sha256(PLAN), "dataset_fingerprint": frozen["fingerprint"],
           "training_addition_fingerprint": release["fingerprint"], "pretoria_sha256": sha256(manifest),
           "reference_checkpoint_sha256": field_plan["checkpoint_sha256"],
           "training_code_sha256": sha256(Path(__file__)),
           "checkpoint_loop_sha256": sha256(ROOT / "leafcheck/potato_study.py"), "preprocessing": spec}
    run_path = output / "run.json"
    if run_path.exists():
        if not resume or read(run_path)["run"] != run:
            raise ValueError("Use --resume with the unchanged registered run")
    else:
        atomic_json(run_path, {"run": run, "config": config, "environment": environment(device), "started_utc": utc_now()})

    def sample(records, size):
        return pixels(records, size), torch.tensor([corn.index(r["class_name"]) for r in records])

    original = Predictor(ARTIFACT, device)
    columns = [original.classes.index(label) for label in corn]
    incumbent = {key: score_pixels(original.scoring_model, sample(r, original.spec["input_size"]),
                                    original.spec, corn, device, config["batch_size"], columns)
                 for key, r in development.items()}
    with_supported_average(incumbent["field"])
    validation = {key: sample(r, config["input_size"]) for key, r in development.items()}

    def assess(scorer):
        result = {key: score_pixels(scorer, samples, spec, corn, device, config["batch_size"])
                  for key, samples in validation.items()}
        with_supported_average(result["field"])
        return result

    all_rows = base + addition
    raw, labels = sample(all_rows, config["input_size"])
    indexes = {r["image_id"]: i for i, r in enumerate(all_rows)}
    seed = config["seed"]
    control = matched_training_rows(base, addition, corn, seed)
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    results = {}
    print("Training", len(base), "+", len(addition), "with date-recorded field validation",
          len(new_dev), "and master", len(development["master"]), flush=True)
    for arm, membership in (("control", control), ("field", all_rows)):
        selected_rows = [indexes[r["image_id"]] for r in membership]
        result = run_arm(output / arm, config, arm, seed, (raw[selected_rows], labels[selected_rows]),
                         validation, incumbent, spec, device, stop, run, assess=assess)
        if result is None or stop["requested"]:
            return
        results[arm] = result
    rules = config["selection"]
    field = results["field"]["scores"]["field"]["macro_f1"]
    control = results["control"]["scores"]["field"]["macro_f1"]
    reference = incumbent["field"]["macro_f1"]
    passed = (results["field"]["passed_master_limits"]
              and field - control >= rules["field_min_gain_vs_control"]
              and field - reference >= rules["field_min_gain_vs_original_given_corn"])
    atomic_json(output / "decision.json", {
        "run": run, "completed_utc": utc_now(), "masked_incumbent": incumbent, "arms": results,
        "training": {"master": len(base), "supplement": release["train_only_counts"]},
        "field_gain_vs_control": field - control, "field_gain_vs_original_given_corn": field - reference,
        "passed_screen": passed, "test_used": False, "deployment_changed": False,
    })
    print("Field gain vs matched control", field - control, "and vs original given corn", field - reference,
          "Passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args.output, args.resume)
