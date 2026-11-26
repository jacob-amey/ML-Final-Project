"""Compare two scratch training arms on the frozen validation and a distinct rust source."""

import argparse
from collections import defaultdict
import math
from pathlib import Path

import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .data import PixelCache, release
from .inference import Predictor
from .scoring import PredictionScores
from .train import device_for, evaluate
from .upload_study import ARTIFACT, read


PLAN = ROOT / "configs/leafcheck_bean_rust_study.json"
RAW = ROOT / "data/raw/leafcheck_field_v12/tanzania-bean"
ARMS = ("matched_repeat", "new_field")
LABEL = "bean__rust"
BEAN_LABELS = ("bean__halo_blight", "bean__mosaic_virus", LABEL)
POLICY = {"name": "d4_probability_mean"}
ORIGINAL_DIAGNOSTIC = ROOT / "docs/field_optimization/tanzania_bean_uploads.json"


def registered_members(plan, frozen):
    tracked = {}
    for name in ("training_edition", "field_validation"):
        path = ROOT / plan[name]["path"]
        if sha256(path) != plan[name]["sha256"]:
            raise ValueError(f"{name} membership changed after registration")
        tracked[name] = read(path)
    training, field = tracked["training_edition"], tracked["field_validation"]
    if (field["root_fingerprint"] != frozen["fingerprint"] or training["base_fingerprint"] != frozen["fingerprint"]
            or field["training_allowed"] or field["makerere_training_manifest_sha256"] != plan["training_edition"]["sha256"]):
        raise ValueError("Study and frozen dataset fingerprints disagree")
    selected = [row for row in field["records"] if row["disposition"] == "validation"]
    if len(selected) != field["roles"]["validation"] or len({r["sha256"] for r in selected}) != len(selected):
        raise ValueError("Field validation has unexpected or duplicate membership")
    if set(r["sha256"] for r in selected).intersection(r["sha256"] for r in training["members"]):
        raise ValueError("Field validation overlaps the treatment training records")
    for row in selected:
        path = RAW / (row["sha256"] + ".jpg")
        if sha256(path) != row["sha256"] or row["label"] != "rust":
            raise ValueError(f"Changed field validation sample: {row['member']}")
    return selected, len(training["members"])


def arm_receipt(run, name, plan, frozen, added):
    config = read(ROOT / "configs" / f"leafcheck_bean_rust_{name}.json")
    expected = {"arm": name, "manifest": plan["training_edition"]["path"], "sha256": plan["training_edition"]["sha256"]}
    if (config["training_addition"] != expected or config["seed"] != plan["seed"]
            or config["input_size"] != plan["input_size"] or config["max_epochs"] != plan["epochs"]):
        raise ValueError(f"The configuration for {name} differs from registrations")
    folder = run / name
    status = read(folder / "status.json")
    if (status["state"] != "complete" or not status["checkpoint_reload_verified"]
            or status["completed_epochs"] != plan["epochs"] or read(folder / "config.json") != config
            or read(folder / "dataset.json")["fingerprint"] != frozen["fingerprint"]
            or status["total_steps"] != plan["epochs"] * math.ceil((59008 + added) / config["batch_size"])):
        raise ValueError(f"Incomplete or mismatched training arm: {name}")
    return status


def field_predictions(predictor, members):
    outcome = []
    for row in members:
        result = predictor.predict(RAW / (row["sha256"] + ".jpg"))
        label = result["label"]
        outcome.append({"member": row["member"], "sha256": row["sha256"], "capture_date_group": row["group"],
                        "reference_label": LABEL, "predicted_label": label, "score": result["score"],
                        "correct": label == LABEL, "wrong_crop": not label.startswith("bean__")})
    return outcome


def field_summary(rows):
    days = defaultdict(list)
    for row in rows:
        days[row["capture_date_group"]].append(row)
    groups = {group: {"support": len(items), "correct": sum(item["correct"] for item in items)}
              for group, items in sorted(days.items())}
    representatives = [min(items, key=lambda row: row["sha256"]) for items in days.values()]
    correct = sum(row["correct"] for row in rows)
    return {"support": len(rows), "correct": correct,
            "wrong_crop": sum(row["wrong_crop"] for row in rows), "rust_recall": correct / len(rows),
            "date_groups": groups,
            "equal_date_recall": sum(group["correct"] / group["support"] for group in groups.values()) / len(groups),
            "lowest_byte_hash_per_date": {"support": len(representatives),
                                          "correct": sum(row["correct"] for row in representatives)}}


def original_nonrust_diagnostic(predictor):
    members = read(ORIGINAL_DIAGNOSTIC)["records"]
    outcomes = {}
    for label in ("healthy", "anthracnose"):
        selected = [row for row in members if row["disposition"] == "diagnostic" and row["label"] == label]
        scores = []
        for row in selected:
            path = RAW / (row["sha256"] + ".jpg")
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Changed original preflight sample: {row['member']}")
            scores.append(predictor.predict(path)["label"] == LABEL)
        outcomes[label] = {"support": len(selected), "predicted_bean_rust": sum(scores)}
    return outcomes


def compare(run, output):
    output.mkdir(parents=True, exist_ok=True)
    result_file = output / "result.json"
    if result_file.exists():
        raise FileExistsError(result_file)
    torch.set_num_threads(4)
    device = device_for("auto")
    frozen, plan = release(), read(PLAN)
    members, added = registered_members(plan, frozen)
    reference = read(ARTIFACT / "freeze.json")
    if sha256(ARTIFACT / "model.pt") != reference["checkpoint_sha256"]:
        raise ValueError("The original deployment checkpoint differs from its freeze")
    receipts = {name: arm_receipt(run, name, plan, frozen, added) for name in ARMS}
    paths = {"deployed": ARTIFACT, **{name: run / name / "best_model.pt" for name in ARMS}}
    cache = PixelCache("val", plan["input_size"], frozen)
    results = {}
    for name, path in paths.items():
        predictor = Predictor(path, device)
        predictor.scoring_model = PredictionScores(predictor.model, POLICY).to(device).eval()
        recorded_validation = None
        if name in ARMS:
            status = receipts[name]
            if predictor.metadata["selected_epoch"] != status["best"]["epoch"]:
                raise ValueError(f"Selected checkpoint no longer matches the receipt for {name}")
            validation, _ = evaluate(predictor.scoring_model, cache, predictor.spec, device, predictor.classes, 128)
            recorded_validation = {key: validation[key] for key in ("sample_count", "correct", "macro_f1", "accuracy", "loss")}
            recorded_validation["bean_classes"] = {v["class_name"]: v for v in validation["per_class"]
                                                    if v["class_name"] in BEAN_LABELS}
            matrix = validation["confusion_matrix"]
            mapping = frozen["class_to_idx"]
            recorded_validation["other_bean_labels_correct"] = sum(matrix[mapping[label]][mapping[label]]
                                                                    for label in BEAN_LABELS[:2])
        rows = field_predictions(predictor, members)
        write_csv(output / f"{name}-field-predictions.csv", rows)
        field = field_summary(rows)
        results[name] = {"checkpoint_sha256": sha256(predictor.path), "epoch": predictor.metadata["selected_epoch"],
                         "resolution": predictor.spec["input_size"], "field_validation": field,
                         "already_seen_nonrust_diagnostic": original_nonrust_diagnostic(predictor),
                         "master_validation": recorded_validation, "training_terminal_state": receipts.get(name)}
        print(name, f"rust {field['correct']}/{field['support']}", f"wrong crop {field['wrong_crop']}",
              "master macro F1", recorded_validation and recorded_validation["macro_f1"], flush=True)

    control, treatment = (results[name] for name in ARMS)
    changes = {k: treatment["master_validation"][k] - control["master_validation"][k]
               for k in ("macro_f1", "accuracy", "other_bean_labels_correct")}
    changes["rust_recall"] = treatment["field_validation"]["rust_recall"] - control["field_validation"]["rust_recall"]
    limit = plan["advance_if_all"]
    passed = (changes["macro_f1"] >= limit["master_macro_f1_delta_at_least"]
              and changes["accuracy"] >= limit["master_accuracy_delta_at_least"]
              and changes["rust_recall"] >= limit["tanzania_rust_recall_delta_at_least"]
              and changes["other_bean_labels_correct"] >= -limit["master_halo_blight_and_mosaic_virus_correct_drop_at_most"])
    atomic_json(result_file, {"measured_utc": utc_now(), "device": str(device), "plan_sha256": sha256(PLAN),
                              "dataset_fingerprint": frozen["fingerprint"], "training_addition": plan["training_edition"],
                              "field_validation": plan["field_validation"], "results": results, "deltas": changes,
                              "descriptive_nonrust_diagnostic_sha256": sha256(ORIGINAL_DIAGNOSTIC),
                              "advance": passed, "deployment_changed": False, "test_opened": False})
    print("Changes", changes, "advance", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.run, args.output)
