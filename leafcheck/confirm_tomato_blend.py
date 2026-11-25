"""Run the frozen one-time known-tomato confirmation on PlantCity."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .inference import Predictor
from .potato_policy import distribution
from .potato_study import verify_arm
from .scoring import PredictionScores
from .tomato_condition_blend import PLAN
from .tomato_study import conditioned, evaluate
from .upload_study import ARTIFACT, network, pixels, read, rows


NAME = "leafcheck_plantcity_holdout_v1"
DATA = ROOT / "data/processed" / NAME
DATA_SEAL = ROOT / "data/manifests" / f"{NAME}_release.json"


def freeze(study, selection, output):
    if (output / "freeze.json").exists():
        raise FileExistsError(output / "freeze.json")
    decision = read(selection)
    chosen = decision["selected_for_confirmation"]
    run = read(study / "run.json")["run"]
    if (not chosen or not decision["validation_only"] or decision["protocol_sha256"] != sha256(PLAN)
            or decision["matched_study_sha256"] != sha256(study / "conditional.json")):
        raise ValueError("A passing selection from the registered development protocol is required")
    if (sha256(Path(chosen["checkpoint"])) != chosen["sha256"]
            or run["reference_checkpoint"] != decision["reference_checkpoint"]):
        raise ValueError("Model fingerprint changed")
    freeze_doc = {
        "created_utc": utc_now(), "dataset_fingerprint": read(DATA_SEAL)["fingerprint"],
        "selection": str(selection), "selection_sha256": sha256(selection), "confirmed_candidate": chosen,
        "run": str(study / "run.json"), "run_sha256": sha256(study / "run.json"), "protocol_sha256": sha256(PLAN),
        "reference_checkpoint": run["reference_checkpoint"], "evaluator_sha256": sha256(Path(__file__)),
        "known_tomato": {"min_macro_f1_gain": 0.03, "max_accuracy_drop": 0.01, "max_recall_drop": 0.05,
                         "guard_labels": ["tomato__healthy", "tomato__early_blight", "tomato__late_blight"]},
        "scope": "Compare both known-tomato methods on the same tomato rows, with all ten tomato conditions "
                 "eligible. Report original automatic predictions separately on the entire collection.",
    }
    atomic_json(output / "freeze.json", freeze_doc)


def score_once(output):
    result_file = output / "metrics.json"
    record_file = output / "predictions.csv"
    if result_file.exists() or record_file.exists():
        raise FileExistsError("Preserve the opened confirmation")
    plan = read(output / "freeze.json")
    run_file = Path(plan["run"])
    selected = plan["confirmed_candidate"]
    checkpoint = Path(selected["checkpoint"])
    for file, key in ((PLAN, "protocol_sha256"), (run_file, "run_sha256"),
                      (Path(plan["selection"]), "selection_sha256"), (Path(__file__), "evaluator_sha256")):
        if sha256(file) != plan[key]:
            raise ValueError(f"Frozen input changed: {file}")
    seal = read(DATA_SEAL)
    if (seal != read(DATA / "release.json") or seal["fingerprint"] != plan["dataset_fingerprint"]
            or any(sha256(DATA / name) != digest for name, digest in seal["files_sha256"].items())):
        raise ValueError("Confirmation dataset changed")
    run = read(run_file)["run"]
    cfg = run["config"]
    if sha256(checkpoint) != selected["sha256"]:
        raise ValueError("Selected specialist changed")
    state = verify_arm(checkpoint, cfg, run)
    torch.set_num_threads(4)
    reference = Predictor(ARTIFACT, "cpu")
    if reference.metadata["checkpoint_sha256"] != plan["reference_checkpoint"]:
        raise ValueError("Reference checkpoint changed")
    classes = reference.classes
    condition_labels = cfg["labels"]
    mapping = {label: idx for idx, label in enumerate(classes)}
    condition_index = {label: idx for idx, label in enumerate(condition_labels)}
    entries = [dict(r, path=DATA / r["output_relative_path"]) for r in rows(DATA / "manifest.csv")
               if r["status"] == "included"]
    support = Counter(row["class_name"] for row in entries)
    if (support != seal["test_only_counts"] or len({r["image_id"] for r in entries}) != len(entries)
            or any(r["split"] != "test" or mapping[r["class_name"]] != int(r["class_id"]) for r in entries)):
        raise ValueError("Unexpected reference labels or repeated images")
    scope = np.flatnonzero([r["class_name"] in condition_index for r in entries])
    column = [mapping[label] for label in condition_labels]

    device = reference.device
    raw = pixels(entries, reference.spec["input_size"])
    base = distribution(reference.scoring_model, raw, reference.spec, device, 64)
    known = conditioned(base[scope], column)
    spec = run["spec"]
    model = network(cfg, device)
    model.load_state_dict(state["weights"], strict=True)
    d4 = PredictionScores(model, {"name": "d4_probability_mean"}).eval()
    extra = distribution(d4, pixels([entries[i] for i in scope], spec["input_size"]), spec, device, 64)
    fraction = selected["fraction"]
    blended = (1 - fraction) * known + fraction * extra
    truth = np.asarray([mapping[r["class_name"]] for r in entries])
    scoped_truth = np.asarray([condition_index[entries[i]["class_name"]] for i in scope])
    automatic = evaluate(base, truth, classes)
    old = evaluate(known, scoped_truth, condition_labels)
    new = evaluate(blended, scoped_truth, condition_labels)
    for result in (automatic, old, new):
        result.pop("confusion_matrix")
    by_class = lambda result: {row["class_name"]: row for row in result["per_class"]}
    guards = plan["known_tomato"]
    losses = {name: by_class(old)[name]["recall"] - by_class(new)[name]["recall"] for name in guards["guard_labels"]}
    support_str = {entry["image_id"]: index for index, entry in enumerate(entries)}
    scoped_pred = {entries[i]["image_id"]: (condition_labels[a], condition_labels[b])
                   for i, a, b in zip(scope, known.argmax(1), blended.argmax(1))}
    records = [
        dict(image_id=row["image_id"], reference=row["class_name"], split=row["split"],
             automatic=classes[base[support_str[row["image_id"]]].argmax()],
             known_reference=scoped_pred.get(row["image_id"], ("", ""))[0],
             known_blend=scoped_pred.get(row["image_id"], ("", ""))[1])
        for row in entries
    ]
    stage = record_file.with_suffix(".tmp")
    write_csv(stage, records)
    stage.replace(record_file)
    gain = new["macro_f1"] - old["macro_f1"]
    accuracy_loss = old["accuracy"] - new["accuracy"]
    passed = (gain >= guards["min_macro_f1_gain"] and accuracy_loss <= guards["max_accuracy_drop"]
              and max(losses.values()) <= guards["max_recall_drop"])
    report = {
        "completed_utc": utc_now(), "freeze_sha256": sha256(output / "freeze.json"),
        "predictions_sha256": sha256(record_file), "automatic_reference": automatic,
        "given_crop_reference": old, "given_crop_blend": new, "known_tomato_macro_f1_gain": gain,
        "known_tomato_accuracy_loss": accuracy_loss, "guard_recall_losses": losses, "passed_confirmation": passed,
    }
    atomic_json(result_file, report)
    print("Automatic correct", automatic["correct"], "of", automatic["sample_count"],
          "known tomato", old["correct"], "to", new["correct"], "of", new["sample_count"],
          "F1 gain", gain, "passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--study", type=Path, default=Path("runs/field_v5/tomato_specialist"))
    parser.add_argument("--selection", type=Path, default=Path("runs/field_v5/tomato_condition_blend.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "freeze":
            freeze(args.study, args.selection, args.output)
        else:
            score_once(args.output)
