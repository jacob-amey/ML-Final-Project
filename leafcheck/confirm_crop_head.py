"""Freeze a repeated crop-head candidate, then score the independent check once."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .crop_head_study import PLAN, blend_crop, measures
from .inference import Predictor
from .potato_policy import distribution
from .potato_study import verify_arm
from .train import device_for
from .upload_study import ARTIFACT, network, pixels, read, rows


CONFIG = ROOT / "configs/leafcheck_crop_head_confirmation_v1.json"
NAME = "leafcheck_crop_confirmation_v1"
DATA = ROOT / "data/processed" / NAME
RECEIPT = ROOT / f"data/manifests/{NAME}_release.json"


def confirmation_release():
    seal = read(RECEIPT)
    actual = {name: sha256(DATA / name) for name in seal["files_sha256"]}
    fingerprint = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
    if read(DATA / "release.json") != seal or actual != seal["files_sha256"] or fingerprint != seal["fingerprint"]:
        raise ValueError("The confirmation manifest differs from its pinned release")
    return seal


def freeze(args):
    plan = read(PLAN)
    results = [read(p) for p in args.decisions]
    expected = plan["seed_plan"]
    if [r["run"]["seed"] for r in results] != expected or any(
        not r["passed_screen"] or r["test_opened"] for r in results
    ):
        raise ValueError("The registered matched repeat must pass before confirmation")
    for result in results:
        run = result["run"]
        if any(run["config"][key] != value for key, value in plan.items()):
            raise ValueError("Trial configuration differs from the registered screen")
        for arm in result["arms"].values():
            if not arm["checkpoint_reload_verified"] or arm["completed_epochs"] != plan["max_epochs"]:
                raise ValueError("The trial requires two completed, verified training arms per seed")
            if sha256(Path(arm["checkpoint"])) != arm["sha256"]:
                raise ValueError("Trial checkpoint changed")
    chosen = max(results, key=lambda r: (r["arms"]["field"]["scores"]["field"]["macro_f1"], -r["run"]["seed"]))
    decision_path = args.decisions[expected.index(chosen["run"]["seed"])]
    arm = chosen["arms"]["field"]
    release = confirmation_release()
    if release["base_release"] != plan["master_fingerprint"]:
        raise ValueError("Confirmation references another master release")
    document = {
        "created_utc": utc_now(),
        "protocol_sha256": sha256(CONFIG),
        "plan_sha256": sha256(PLAN),
        "evaluator_sha256": sha256(Path(__file__)),
        "blend_implementation_sha256": sha256(Path(__file__).with_name("crop_head_study.py")),
        "confirmatory_data": release["fingerprint"],
        "reference_checkpoint": plan["reference_checkpoint"],
        "candidate_checkpoint": arm["checkpoint"],
        "candidate_sha256": arm["sha256"],
        "candidate_seed": arm["seed"],
        "decisions": {str(path): sha256(path) for path in args.decisions},
        "selected_decision": str(decision_path),
        "reference_class_map": sha256(ARTIFACT / "class_to_idx.json"),
    }
    atomic_json(args.output / "freeze.json", document)
    print("Frozen crop-head candidate", arm["seed"], arm["sha256"], flush=True)


def confirm(out):
    freeze_path, prediction_path, metrics_path = (out / n for n in ("freeze.json", "predictions.csv", "metrics.json"))
    if prediction_path.exists() or metrics_path.exists():
        raise FileExistsError("Preserve the already opened confirmation. Reconcile the saved results.")
    record = read(freeze_path)
    seal = confirmation_release()
    if (record["protocol_sha256"] != sha256(CONFIG) or record["plan_sha256"] != sha256(PLAN)
            or record["evaluator_sha256"] != sha256(Path(__file__))
            or record["blend_implementation_sha256"] != sha256(Path(__file__).with_name("crop_head_study.py"))
            or seal["fingerprint"] != record["confirmatory_data"]
            or any(sha256(Path(path)) != value for path, value in record["decisions"].items())
            or sha256(ARTIFACT / "class_to_idx.json") != record["reference_class_map"]
            or sha256(Path(record["candidate_checkpoint"])) != record["candidate_sha256"]):
        raise ValueError("Candidate, plan, class map or confirmation release changed after freeze")
    torch.set_num_threads(4)
    device = device_for("auto")
    predictor = Predictor(ARTIFACT, device)
    if predictor.metadata["checkpoint_sha256"] != record["reference_checkpoint"]:
        raise ValueError("Reference checkpoint changed after freeze")
    run = read(Path(record["selected_decision"]))["run"]
    config = run["config"]
    model = network(config, device)
    state = verify_arm(Path(record["candidate_checkpoint"]), config, run)
    model.load_state_dict(state["weights"], strict=True)
    class_names = predictor.classes
    crop_index = np.array([config["labels"].index(label.split("__", 1)[0]) for label in class_names])
    members = [dict(r, path=DATA / r["output_relative_path"])
               for r in rows(DATA / "manifest.csv") if r["status"] == "included" and r["split"] == "test"]
    included = Counter(r["class_name"] for r in members)
    expected = read(CONFIG)
    if (included != seal["test_only_counts"] or min(included.values()) < expected["min_examples_per_label"]
            or set(r["source"] for r in members) != set(expected["expected_sources"])
            or len({r["image_id"] for r in members}) != len(members)
            or any(class_names[int(r["class_id"])] != r["class_name"] for r in members)):
        raise ValueError("The registered independent class or source support changed")
    results = {}
    predictions = []
    recall_losses = {}
    gains = {}
    crop_reductions = {}
    for source in expected["expected_sources"]:
        subset = [r for r in members if r["source"] == source]
        labels = np.array([int(r["class_id"]) for r in subset])
        raw = pixels(subset, predictor.spec["input_size"])
        base = distribution(predictor.scoring_model, raw, predictor.spec, device)
        crop = distribution(model, pixels(subset, run["crop_spec"]["input_size"]), run["crop_spec"], device)
        candidate = blend_crop(base, crop, crop_index, config["crop_blend"])
        left, right = measures(base, labels, class_names), measures(candidate, labels, class_names)
        results[source] = {"reference": left, "candidate": right}
        gains[source] = right["macro_f1"] - left["macro_f1"]
        crop_reductions[source] = (left["wrong_crop"] - right["wrong_crop"]) / len(subset)
        for before, after in zip(left["per_class"], right["per_class"], strict=True):
            assert before["class_name"] == after["class_name"]
            recall_losses[before["class_name"]] = before["recall"] - after["recall"]
        for i, r in enumerate(subset):
            before, after = (int(distribution[i].argmax()) for distribution in (base, candidate))
            predictions.append({
                "image_id": r["image_id"], "source": source, "group_id": r["group_id"], "actual": r["class_name"],
                "reference": class_names[before], "candidate": class_names[after],
                "reference_score": float(base[i, before]), "candidate_score": float(candidate[i, after]),
            })
    main_gain = float(np.mean(list(gains.values())))
    crop_reduction = float(np.mean(list(crop_reductions.values())))
    passed = (
        main_gain >= expected["min_mean_macro_f1_gain"]
        and crop_reduction >= expected["min_mean_crop_error_reduction"]
        and min(gains.values()) >= -expected["max_source_macro_f1_drop"]
        and max(recall_losses.values()) <= expected["max_condition_recall_drop"]
    )
    write_csv(prediction_path, predictions)
    atomic_json(metrics_path, {
        "completed_utc": utc_now(), "freeze_sha256": sha256(freeze_path), "prediction_sha256": sha256(prediction_path),
        "device": str(device), "results": results, "source_macro_f1_gains": gains, "mean_macro_f1_gain": main_gain,
        "mean_crop_error_reduction": crop_reduction, "recall_losses": recall_losses, "passed_confirmation": passed,
    })
    print("Independent crop confirmation:", gains, "mean crop error reduction", crop_reduction, "passed", passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, nargs=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "freeze":
            if not args.decisions or any((args.output / file).exists() for file in ("freeze.json", "predictions.csv", "metrics.json")):
                raise ValueError("Freeze once with both matched development decisions")
            freeze(args)
        else:
            confirm(args.output)
