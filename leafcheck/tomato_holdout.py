"""Evaluate fixed automatic and known-tomato predictions on one reserved source."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import subprocess

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .inference import Predictor
from .potato_policy import describe_set, distribution, second_member
from .upload_study import ARTIFACT, load_study, pixels, read, rows, score


PLAN = ROOT / "configs/leafcheck_bangladesh_tomato_v1_eval.json"
RELEASE = ROOT / "data/processed/leafcheck_bangladesh_tomato_v1"
VALIDATION = ROOT / "docs/potato_context/validation.json"


def check_inputs(release, plan):
    hashes = {
        "model.pt": plan["reference_sha256"],
        "class_to_idx.json": plan["class_map_sha256"],
        "preprocessing.json": plan["preprocessing_sha256"],
    }
    if any(sha256(ARTIFACT / name) != expected for name, expected in hashes.items()):
        raise ValueError("Frozen model files differ from the evaluation plan")
    if sha256(ROOT / plan["secondary"]["member"]) != plan["secondary"]["member_sha256"]:
        raise ValueError("Frozen secondary checkpoint differs from the evaluation plan")
    seal = read(release / "release.json")
    if seal["base_release"] != plan["master_fingerprint"] or any(
        sha256(release / name) != expected for name, expected in seal["files_sha256"].items()
    ):
        raise ValueError("Source screening receipt does not match its manifest")
    return seal


def freeze(output, release):
    path = output / "freeze.json"
    if path.exists():
        raise FileExistsError(path)
    plan = read(PLAN)
    seal = check_inputs(release, plan)
    validation = read(VALIDATION)
    registered_member = validation["plan"]["automatic_ensemble"]
    if not validation["automatic"]["eligible_on_validation"] or any(
        registered_member[key] != value for key, value in plan["secondary"].items()
    ):
        raise ValueError("The ensemble does not match the completed validation decision")
    atomic_json(path, {"config": plan, "release": seal, "frozen_utc": utc_now(),
                       "evaluator_sha256": sha256(Path(__file__)),
                       "validation_decision_sha256": sha256(VALIDATION),
                       "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})


def evaluate(output, release):
    prediction_path = output / "predictions.csv"
    if (output / "metrics.json").exists() or prediction_path.exists():
        raise FileExistsError("Preserving existing evaluation outputs")
    freeze_path = output / "freeze.json"
    frozen = read(freeze_path)
    plan = frozen["config"]
    if check_inputs(release, plan) != frozen["release"] or sha256(Path(__file__)) != frozen["evaluator_sha256"]:
        raise ValueError("Frozen release or evaluator changed")
    torch.set_num_threads(4)
    original = Predictor(ARTIFACT, "cpu")
    if original.policy["name"] != plan["policy"]:
        raise ValueError("Unexpected scoring policy")
    alternative = second_member(plan["secondary"], original, original.device)
    classes = original.classes
    represented = [classes.index(label) for label in plan["represented_labels"]]
    tomato_labels = np.flatnonzero([label.startswith(plan["known_crop"] + "__") for label in classes])
    entries = [dict(row, path=release / row["output_relative_path"]) for row in rows(release / "manifest.csv")
               if row["status"] == "included"]
    manifest_counts = Counter(row["class_name"] for row in entries)
    if (set(manifest_counts) != set(plan["represented_labels"])
            or manifest_counts != frozen["release"]["test_only_counts"]
            or len({row["image_id"] for row in entries}) != len(entries)
            or any(row["split"] != "test" or int(row["class_id"]) != classes.index(row["class_name"]) for row in entries)):
        raise ValueError("Release has unexpected labels, duplicate IDs, or mixed roles")
    y = describe_set(entries, classes)
    raw = pixels(entries, original.spec["input_size"])
    a = distribution(original.scoring_model, raw, original.spec, original.device)
    b = distribution(alternative.scoring_model, raw, alternative.spec, alternative.device)
    ensemble = .5 * (a + b)
    _, master, _ = load_study(read(ROOT / "configs/upload_blight_v1.json"))
    counts = Counter(row["class_name"] for row in master if row["split"] == "train")

    def rule(indices):
        return max(indices, key=lambda i: (counts[classes[i]], -i))

    automatic = rule(range(len(classes)))
    given_tomato = rule(tomato_labels)
    predictions = {
        "automatic_original": a.argmax(1),
        "automatic_two_seed": ensemble.argmax(1),
        "known_tomato_original": tomato_labels[a[:, tomato_labels].argmax(1)],
        "known_tomato_two_seed": tomato_labels[ensemble[:, tomato_labels].argmax(1)],
        "automatic_frequency_rule": np.full(len(entries), automatic),
        "known_tomato_frequency_rule": np.full(len(entries), given_tomato),
    }
    metrics = {}
    for name, result in predictions.items():
        summary = score(y, result, classes, represented)
        summary.pop("confusion_matrix")
        summary["per_class"] = [summary["per_class"][i] for i in represented]
        metrics[name] = summary
    records = [dict(image_id=row["image_id"], reference=row["class_name"], path=row["output_relative_path"],
                    **{name: classes[result[i]] for name, result in predictions.items()}) for i, row in enumerate(entries)]
    staged = prediction_path.with_suffix(".csv.tmp")
    write_csv(staged, records)
    staged.replace(prediction_path)
    atomic_json(output / "metrics.json", {"methods": metrics, "evaluated_utc": utc_now(),
                                          "manifest_counts": dict(manifest_counts),
                                          "rules": {"automatic": classes[automatic], "known_tomato": classes[given_tomato]},
                                          "freeze_sha256": sha256(freeze_path),
                                          "predictions_sha256": sha256(prediction_path)})
    print({name: (m["correct"], m["sample_count"], m["macro_f1"], m["wrong_crop"])
           for name, m in metrics.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=RELEASE)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (freeze if args.action == "freeze" else evaluate)(args.output, args.release)
