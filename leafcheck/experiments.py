"""Summarize actual development runs and the validation simple rule."""

import csv
import json
from pathlib import Path
import numpy as np
from .common import ROOT, RUNS, atomic_json, metrics_from_matrix, write_csv
from .data import release, split_rows


def main():
    frozen = release(); classes = frozen["classes"]
    decisions_path = ROOT / "docs/experiment_decisions.json"
    decisions = json.loads(decisions_path.read_text()) if decisions_path.exists() else {}
    records = []
    for directory in sorted(RUNS.iterdir()):
        if not all((directory / name).exists() for name in ("history.csv", "config.json", "status.json")): continue
        config = json.loads((directory / "config.json").read_text())
        history = list(csv.DictReader((directory / "history.csv").open()))
        if not history: continue
        status = json.loads((directory / "status.json").read_text())
        environment = json.loads((directory / "environment.json").read_text())
        best = status["best"]
        records.append({"run": directory.name, "experiment": config["experiment"], "state": status["state"],
                        "architecture": config["model"]["architecture"], "channels": str(config["model"]["channels"]),
                        "input_size": config["input_size"], "seed": config["seed"], "class_weight_power": config.get("class_weight_power", 0),
                        "completed_epochs": len(history), "selected_epoch": best["epoch"],
                        "validation_macro_f1": best["macro_f1"], "validation_accuracy": best["accuracy"], "validation_loss": best["loss"],
                        "training_seconds": status.get("training_seconds", sum(float(r["epoch_seconds"]) for r in history)),
                        "additional_refinement_seconds": status.get("additional_refinement_seconds", ""),
                        "parent_run": status.get("parent_run", ""),
                        "hypothesis": config["hypothesis"], "change_from_reference": config.get("reference"),
                        "dataset_fingerprint": frozen["fingerprint"], "git_revision": environment["git_revision"],
                        "source_fingerprint": environment["source_fingerprint"],
                        "decision": decisions.get(directory.name, {}).get("decision", "Pending; run/selection incomplete"),
                        "rationale": decisions.get(directory.name, {}).get("rationale", "")})
    if records: write_csv(ROOT / "docs/experiments.csv", records)
    rule = json.loads((ROOT / "data/manifests/leafcheck177_rule.json").read_text())
    matrix = np.zeros((177, 177), np.int64)
    counts = np.bincount([int(row["class_id"]) for row in split_rows("val")], minlength=177)
    matrix[:, rule["class_index"]] = counts
    result = metrics_from_matrix(matrix, classes)
    summary = {key: result[key] for key in ["correct", "sample_count", "accuracy", "macro_f1"]}
    atomic_json(ROOT / "docs/validation_rule.json", {"rule": rule, "validation": summary})
    print(json.dumps({"runs": len(records), "validation_rule": summary}, indent=2))


if __name__ == "__main__": main()
