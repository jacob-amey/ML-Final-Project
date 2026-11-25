"""Check a fixed list of condition blends on development only."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now
from .inference import Predictor
from .potato_policy import distribution
from .potato_study import verify_arm
from .scoring import PredictionScores
from .tomato_study import aggregate, conditioned, data_for, evaluate
from .train import device_for
from .upload_study import ARTIFACT, network, pixels, read, summarize


PLAN = ROOT / "configs/leafcheck_tomato_condition_blend_v1.json"


def run(study, output):
    if output.exists():
        raise FileExistsError(output)
    run_receipt = read(study / "run.json")["run"]
    config = run_receipt["config"]
    if not (study / "joint.json").exists():
        raise ValueError("Finish the matched study before evaluating blended scores")
    experiment_file = study / "conditional.json"
    experiment = read(experiment_file)
    rule = read(PLAN)
    torch.set_num_threads(4)
    device = device_for("auto")
    _, _, _, _, dev = data_for(config)
    classes = config["labels"]
    baseline = Predictor(ARTIFACT, device)
    spec = run_receipt["spec"]
    if baseline.metadata["checkpoint_sha256"] != run_receipt["reference_checkpoint"]:
        raise ValueError("Reference checkpoint changed")
    columns = [baseline.classes.index(label) for label in classes]
    target = {key: np.asarray([classes.index(row["class_name"]) for row in records]) for key, records in dev.items()}
    original = {}
    raw = {}
    for key, records in dev.items():
        p = distribution(baseline.scoring_model, pixels(records, baseline.spec["input_size"]),
                         baseline.spec, device, size=config["batch_size"])
        original[key] = conditioned(p, columns)
        raw[key] = pixels(records, spec["input_size"])
    reference = {key: evaluate(p, target[key], classes) for key, p in original.items()}
    source_names = config["selection_sources"]
    reference_f1 = aggregate(reference, source_names)["macro_f1"]
    candidates = {str(fraction): {} for fraction in rule["fractions"]}
    for arm, rows in experiment["arms"].items():
        for row in rows:
            checkpoint = Path(row["checkpoint"])
            if sha256(checkpoint) != row["sha256"]:
                raise ValueError("Selected checkpoint changed")
            state = verify_arm(checkpoint, config, run_receipt)
            model = network(config, device)
            model.load_state_dict(state["weights"], strict=True)
            scorer = PredictionScores(model, {"name": "d4_probability_mean"}).eval()
            alternate = {key: distribution(scorer, data, spec, device, size=config["batch_size"])
                         for key, data in raw.items()}
            for fraction in rule["fractions"]:
                scores = {key: evaluate((1 - fraction) * p + fraction * alternate[key], target[key], classes)
                          for key, p in original.items()}
                row_scores = {"seed": row["seed"], "checkpoint": str(checkpoint), "sha256": row["sha256"],
                              "mean_field_f1": aggregate(scores, source_names)["macro_f1"],
                              "sources": scores}
                candidates[str(fraction)].setdefault(arm, []).append(row_scores)
    passes = []
    for fraction, arms in candidates.items():
        mean = float(np.mean([r["mean_field_f1"] for r in arms["field"]]))
        control = float(np.mean([r["mean_field_f1"] for r in arms["control"]]))
        report = {"fraction": float(fraction), "mean_field_f1": mean, "mean_control_f1": control,
                  "gain_over_reference": mean - reference_f1, "gain_over_control": mean - control}
        guards = all(
            candidate["sources"]["master"]["macro_f1"] >= reference["master"]["macro_f1"] - rule["max_master_macro_f1_drop"]
            and candidate["sources"]["master"]["accuracy"] >= reference["master"]["accuracy"] - rule["max_master_accuracy_drop"]
            and all(candidate["sources"][src]["macro_f1"] >= reference[src]["macro_f1"] - rule["max_development_source_drop"]
                    for src in source_names)
            and candidate["mean_field_f1"] > ctrl["mean_field_f1"]
            for candidate, ctrl in zip(arms["field"], arms["control"])
        )
        report["passes"] = (guards and report["gain_over_reference"] >= rule["min_gain_vs_incumbent"]
                            and report["gain_over_control"] >= rule["min_gain_vs_control"])
        report["master_f1"] = [r["sources"]["master"]["macro_f1"] for r in arms["field"]]
        passes.append(report)
    eligible = sorted((r for r in passes if r["passes"]), key=lambda r: (-r["mean_field_f1"], r["fraction"]))
    chosen = None
    if eligible:
        chosen = eligible[0].copy()
        finalist = max(candidates[str(chosen["fraction"])]["field"], key=lambda r: (r["mean_field_f1"], -r["seed"]))
        chosen.update({key: finalist[key] for key in ("checkpoint", "sha256", "seed")})
    record = {
        "evaluated_utc": utc_now(), "protocol_sha256": sha256(PLAN), "matched_study_sha256": sha256(experiment_file),
        "reference_checkpoint": baseline.metadata["checkpoint_sha256"], "validation_only": True,
        "reference": reference, "results": candidates, "decisions": passes, "selected_for_confirmation": chosen,
    }
    atomic_json(output, record)
    print(passes, "selected", chosen, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.output)
