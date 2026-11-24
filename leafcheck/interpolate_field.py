"""Check whether a smaller move toward the trained weights preserves broad accuracy."""

import argparse
from pathlib import Path

import torch

from .common import atomic_checkpoint, atomic_json, sha256, utc_now
from .data import PixelCache
from .field_expansion import choose, is_eligible, load_inputs, prepare_development, validate
from .inference import Predictor
from .train import cpu_state, device_for
from .upload_study import ARTIFACT, read


def run(release_dir, parent_run, output_dir, plan_file):
    plan = read(plan_file)
    config = read(parent_run / "run.json")["config"]
    primary = read(parent_run / "decision.json")
    if (parent_run / "status.json").exists() and read(parent_run / "status.json")["state"] != "complete":
        raise ValueError("Complete the paired parent experiment first")
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = device_for("auto")
    frozen, release, previous, new = load_inputs(release_dir, config)
    if primary["release"] != release["fingerprint"]:
        raise ValueError("The parent used a different dataset")
    spec = read(ARTIFACT / "preprocessing.json")
    validation = PixelCache("val", spec["input_size"], frozen)
    dev = prepare_development(previous, new, spec["input_size"])
    predictor = Predictor(ARTIFACT, device)
    base = cpu_state(predictor.model)
    baseline = read(parent_run / "baseline.json")
    policy = "d4_probability_mean"
    rules = config["selection"]
    records, final = {}, {}
    atomic_json(output_dir / "receipt.json", {"plan": plan, "parent": sha256(parent_run / "decision.json"),
                                            "release": release["fingerprint"], "utc": utc_now()})
    for arm in ("matched_control", "new_field"):
        parent = primary["arms"][arm]
        path = Path(parent["checkpoint"])
        if sha256(path) != parent["sha256"]:
            raise ValueError("The parent checkpoint changed")
        end = torch.load(path, weights_only=True, map_location="cpu")["weights"]
        favored = None
        selected_state = None
        records[arm] = []
        for alpha in plan["new_weight_fractions"]:
            state = {key: start.lerp(end[key], alpha) if start.is_floating_point() else start
                     for key, start in base.items()}
            predictor.model.load_state_dict(state, strict=True)
            evidence = validate(predictor.model, validation, dev, predictor.classes, spec, device, config, "single_view")
            evidence["eligible"] = is_eligible(evidence, baseline["single_view"], rules)
            evidence["new_weight_fraction"] = alpha
            records[arm].append(evidence)
            atomic_json(output_dir / "screening.json", records)
            print(arm, alpha, evidence["master"]["macro_f1"], evidence["selection_field_f1"], evidence["eligible"], flush=True)
            if favored is None or (evidence["eligible"] and not favored["eligible"]) or (
                    evidence["eligible"] == favored["eligible"] and choose(evidence, favored)):
                favored, selected_state = evidence, state
        predictor.model.load_state_dict(selected_state, strict=True)
        result = validate(predictor.model, validation, dev, predictor.classes, spec, device, config, policy)
        checkpoint = output_dir / f"{arm}.pt"
        atomic_checkpoint(checkpoint, {"weights": cpu_state(predictor.model), "result": result})
        result.update(checkpoint=str(checkpoint), sha256=sha256(checkpoint), epoch=parent["epoch"],
                      new_weight_fraction=favored["new_weight_fraction"])
        final[arm] = result
        atomic_json(output_dir / f"{arm}.json", result)
        print(arm, policy, result["master"]["macro_f1"], result["selection_field_f1"], flush=True)
    candidate = final["new_field"]
    strong_control = max(final["matched_control"]["selection_field_f1"],
                         primary["arms"]["matched_control"]["selection_field_f1"]
                         if is_eligible(primary["arms"]["matched_control"], baseline[policy], rules) else 0.0)
    promote = (is_eligible(candidate, baseline[policy], rules)
               and candidate["selection_field_f1"] >= strong_control + rules["control_min_gain"])
    atomic_json(output_dir / "decision.json", {
        "baseline": baseline[policy], "arms": final, "promotion_guards_passed": promote, "test_opened": False,
        "release": release["fingerprint"], "base_checkpoint": config["initial_weights"], "development_utc": utc_now(),
    })


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--release", required=True, type=Path)
    cli.add_argument("--parent", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--plan", required=True, type=Path)
    args = cli.parse_args()
    run(args.release, args.parent, args.output, args.plan)
