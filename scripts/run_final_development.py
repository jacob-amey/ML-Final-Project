"""Continue the preregistered duration and scoring comparisons after seed 435.

Use the same accelerator lock as the native-resolution comparison. This driver
records an incumbent and recommendation while both holdouts remain sealed.
"""

import csv
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leafcheck.common import atomic_json, sha256, utc_now
from scripts import run_scaling_comparison as workflow

RUNS = ROOT / "runs/final177"
DOCS = ROOT / "docs"
LOG = RUNS / "final_development_tasks.log"


def read(path):
    return json.loads(path.read_text())


def late_evidence(run):
    status = read(run / "status.json")
    config = read(run / "config.json")
    assert status["state"] == "complete" and status["checkpoint_reload_verified"]
    with (run / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == status["completed_epochs"] and all(int(row["train_images"]) == 59008 for row in rows)
    scores = [float(row["validation_macro_f1"]) for row in rows]
    return {"run": run.name, "hit_cap": len(rows) == config["max_epochs"],
            "late_peak": status["best"]["epoch"] > len(rows) - 8,
            "late_gain": max(scores[-8:]) - max(scores[:-8]),
            "next_learning_rate": float(rows[-1]["learning_rate_next"])}


def extension_pair(original):
    paths = []
    for record in original["runs"]:
        parent = RUNS / record["run"]
        output = RUNS / f"residual177_256_long80_seed{record['seed']}"
        assert sha256(parent / "best_model.pt") == record["checkpoint_sha256"]
        if not output.exists():
            workflow.command(["-m", "leafcheck.extend_run", "--parent", str(parent), "--output", str(output), "--max-epochs", "80"], LOG)
        lineage = read(output / "lineage.json")
        config = read(output / "config.json")
        assert lineage["parent_run"] == parent.name and lineage["parent_best_sha256"] == record["checkpoint_sha256"]
        description = {"experiment", "hypothesis", "reference", "max_epochs"}
        assert {k:v for k,v in config.items() if k not in description} == {k:v for k,v in lineage["parent_config"].items() if k not in description}
        assert config["max_epochs"] == 80
        atomic_json(DOCS / "active_training.json", {"state": "paired_duration_trial", "run": str(output.relative_to(ROOT)),
                    "incumbent": "docs/seed_stability_256.json", "updated_utc": utc_now(), "test_predictions_inspected": False})
        workflow.run(output / "config.json", output)
        paths.append(str(output))
    prefix = DOCS / "seed_stability_256_long80"
    workflow.command(["-m", "leafcheck.seed_analysis", "--runs", *paths, "--output", str(prefix)], LOG)
    return prefix.with_suffix(".json")


def main():
    signal.signal(signal.SIGINT, workflow.stop)
    signal.signal(signal.SIGTERM, workflow.stop)
    if (ROOT / "artifacts/leafcheck177-v1/freeze.json").exists():
        raise SystemExit("Model is frozen. Continue the delivery workflow.")
    comparison = read(DOCS / "scaling_256_comparison.json")
    assert comparison["state"] == "complete" and len(comparison["completed_procedures"]) == 2, "Finish or resume the native-size comparison first"
    reference = read(DOCS / "seed_stability_224.json")
    candidate = read(DOCS / "seed_stability_256.json")
    gain = candidate["mean_macro_f1"] - reference["mean_macro_f1"]
    promoted = gain >= .005
    assert promoted == comparison["accepted_by_declared_rule"]
    pair = DOCS / f"seed_stability_{256 if promoted else 224}.json"
    evidence = [late_evidence(RUNS / row["run"]) for row in candidate["runs"]]
    extend = promoted and all(row["hit_cap"] for row in evidence) and any(
        row["late_peak"] and row["late_gain"] >= .003 and row["next_learning_rate"] > .00002 for row in evidence)
    decision = {"recorded_utc": utc_now(), "evaluation_split": "validation only", "native_size_accepted": promoted,
                "native_size_paired_gain": gain, "parent_pair": pair.name, "late_curve_evidence": evidence,
                "duration_trial_triggered": extend, "duration_trial_complete": False, "test_predictions_used": False}
    atomic_json(DOCS / "duration_trial_decision.json", decision)
    print(f"Native resolution comparison complete. Incumbent: {pair.name}. Extend: {extend}.", flush=True)
    if extend:
        challenger = extension_pair(candidate)
        extended = read(challenger)
        gains = [after["macro_f1"] - before["macro_f1"] for before, after in zip(candidate["runs"], extended["runs"])]
        mean_gain = extended["mean_macro_f1"] - candidate["mean_macro_f1"]
        accepted = mean_gain >= .005 and min(gains) >= 0
        decision.update(duration_trial_complete=True, duration_trial_accepted=accepted, duration_paired_mean_gain=mean_gain, duration_seed_gains=gains)
        if accepted:
            pair = challenger
    decision["incumbent_pair"] = pair.name
    atomic_json(DOCS / "duration_trial_decision.json", decision)
    atomic_json(DOCS / "active_training.json", {"state": "validation_inference_comparison", "incumbent": pair.name,
                "updated_utc": utc_now(), "test_predictions_inspected": False})
    workflow.command(["-m", "leafcheck.compare_inference", "--pair", str(pair), "--output", str(RUNS / "inference_comparison")], LOG)
    workflow.command(["-m", "leafcheck.experiments"], LOG)
    policy = read(DOCS / "inference_policy_comparison.json")
    atomic_json(DOCS / "active_training.json", {"state": "development_review_ready", "incumbent": pair.name,
                "selected_run_recommendation": policy["selected_run"], "next_action": "Review saved validation evidence and freeze the deliverable before evaluation.",
                "updated_utc": utc_now(), "test_predictions_inspected": False})
    print("Declared validation comparisons completed. Review before the final freeze.", flush=True)


if __name__ == "__main__":
    main()
