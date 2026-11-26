"""Freeze and score a prospective comparison of automatic and known-tomato inference."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .inference import Predictor
from .potato_policy import distribution
from .tomato_study import conditioned, evaluate
from .upload_study import ARTIFACT, pixels, read, rows


NAME = "leafcheck_hossain_tomato_v1"
DATA = ROOT / "data/processed" / NAME
SEAL = ROOT / f"data/manifests/{NAME}_release.json"
PLAN = ROOT / "configs/leafcheck_tomato_context_v1.json"
DEVELOPMENT = ROOT / "docs/field_optimization/tomato_context_screen.json"
AUDIT = ROOT / "docs/field_optimization/hossain_tomato_audit.json"
CLASSES = {"tomato__early_blight", "tomato__late_blight", "tomato__healthy"}


def checked_inputs():
    plan, evidence, seal, audit = map(read, (PLAN, DEVELOPMENT, SEAL, AUDIT))
    if (not evidence["qualified_for_confirmation"] or not evidence["development_only"]
            or evidence["reference_sha256"] != plan["checkpoint_sha256"]
            or evidence["config_sha256"] != sha256(PLAN)):
        raise ValueError("The context policy needs a matching development decision")
    if (seal != read(DATA / "release.json") or audit["release_fingerprint"] != seal["fingerprint"]
            or not audit["eligible_for_confirmation"]
            or any(sha256(DATA / name) != digest for name, digest in seal["files_sha256"].items())):
        raise ValueError("The screened independent release differs from the registered evidence")
    entries = [dict(r, path=DATA / r["output_relative_path"])
               for r in rows(DATA / "manifest.csv") if r["status"] == "included"]
    counts = Counter(row["class_name"] for row in entries)
    if (set(counts) != CLASSES or counts != seal["test_only_counts"] or min(counts.values()) < 25
            or len({row["image_id"] for row in entries}) != len(entries)
            or len({row["pixel_d4_sha256"] for row in entries}) != len(entries)
            or any(row["split"] != "test" or row["source"] != "hossain" for row in entries)):
        raise ValueError("Independent membership or required class support differs from the plan")
    pins = {
        "protocol_sha256": sha256(PLAN), "development_sha256": sha256(DEVELOPMENT),
        "audit_sha256": sha256(AUDIT), "evaluator_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(ARTIFACT / "model.pt"),
        "class_map_sha256": sha256(ARTIFACT / "class_to_idx.json"),
        "preprocessing_sha256": sha256(ARTIFACT / "preprocessing.json"),
        "artifact_sha256": sha256(ARTIFACT / "artifact.json"),
        "dataset_fingerprint": seal["fingerprint"], "class_support": dict(counts),
    }
    if pins["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("The original checkpoint changed")
    return plan, pins, entries


def freeze(output):
    path = output / "freeze.json"
    if path.exists():
        raise FileExistsError(path)
    _, pins, _ = checked_inputs()
    atomic_json(path, {"created_utc": utc_now(), **pins})


def score(output):
    freeze_file = output / "freeze.json"
    original_freeze = read(freeze_file)
    table, result = output / "predictions.csv", output / "metrics.json"
    if table.exists() or result.exists():
        raise FileExistsError("Preserve the already opened evaluation.")
    plan, pins, entries = checked_inputs()
    if any(original_freeze[key] != expected for key, expected in pins.items()):
        raise ValueError("Frozen context experiment inputs changed")
    torch.set_num_threads(4)
    predictor = Predictor(ARTIFACT, "cpu")
    if predictor.policy["name"] != "d4_probability_mean":
        raise ValueError("Expected the original deployed inference policy")
    classes = predictor.classes
    columns = [i for i, label in enumerate(classes) if label.startswith("tomato__")]
    names = [classes[i] for i in columns]
    if len(names) != 10 or any(classes[int(row["class_id"])] != row["class_name"] for row in entries):
        raise ValueError("Invalid class order or mapping")
    original = distribution(predictor.scoring_model, pixels(entries, predictor.spec["input_size"]), predictor.spec,
                            predictor.device, 64)
    tomato = conditioned(original, columns)
    truth = np.array([int(row["class_id"]) for row in entries])
    known_truth = np.array([names.index(row["class_name"]) for row in entries])
    automatic = evaluate(original, truth, classes)
    context = evaluate(tomato, known_truth, names)
    for item in (automatic, context):
        item.pop("confusion_matrix")
        item["blight"] = [row for row in item["blight"] if row["total"]]
    recall_before = {row["class_name"]: row["recall"] for row in automatic["per_class"]}
    recall_after = {row["class_name"]: row["recall"] for row in context["per_class"]}
    losses = {label: recall_before[label] - recall_after[label] for label in sorted(CLASSES)}
    gains = {name: context[name] - automatic[name] for name in ("accuracy", "macro_f1")}
    rules = plan["confirmation"]
    passed = (gains["macro_f1"] >= rules["min_joint_macro_f1_gain"]
              and gains["accuracy"] >= rules["min_joint_accuracy_gain"]
              and losses["tomato__late_blight"] <= rules["max_late_blight_recall_drop"])
    stage = table.with_suffix(".tmp")
    write_csv(stage, [
        dict(image_id=row["image_id"], group_id=row["group_id"], truth=row["class_name"],
             automatic=classes[j], automatic_score=float(original[i, j]),
             given_tomato=names[k], given_tomato_score=float(tomato[i, k]))
        for i, (row, j, k) in enumerate(zip(entries, original.argmax(1), tomato.argmax(1), strict=True))
    ])
    stage.replace(table)
    atomic_json(result, {
        "completed_utc": utc_now(), "freeze_sha256": sha256(freeze_file), "predictions_sha256": sha256(table),
        "automatic": automatic, "given_tomato": context, "gain": gains, "recall_loss": losses,
        "passed_confirmation": passed, "tomato_labels": names,
    })
    print("Automatic", automatic["correct"], "given tomato", context["correct"], "of", len(entries),
          "gains", gains, "passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (freeze if args.action == "freeze" else score)(args.output)
