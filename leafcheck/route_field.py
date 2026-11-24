"""Measure a validation-only combination of incumbent and field CNN scores."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import atomic_json, sha256
from .data import PixelCache
from .field_expansion import choose, is_eligible, load_inputs, prepare_development
from .inference import Predictor
from .preprocessing import prepare_batch
from .train import device_for
from .upload_study import ARTIFACT, read, score, summarize


@torch.inference_mode()
def collect(predictor, data, batch_size):
    outputs = {}
    for source, (labels, batches) in data.items():
        res = [predictor.scoring_model(prepare_batch(batch, predictor.spec, predictor.device)).softmax(1).cpu()
               for batch in batches(batch_size)]
        outputs[source] = (labels, torch.cat(res).numpy())
    return outputs


def routed(reference, future, cutoff, candidate_min):
    take = (reference.max(1) < cutoff) & (future.max(1) >= candidate_min)
    return np.where(take[:, None], future, reference), int(take.sum())


def quantify(reference, extra, classes, plan, minimum):
    threshold = plan["baseline_score_threshold"]
    result, stats = {}, {}
    for source, (labels, probs) in reference.items():
        if extra is None:
            effective, switches = probs, 0
        else:
            effective, switches = routed(probs, extra[source][1], threshold, minimum)
        predicted = effective.argmax(1)
        selected = effective.max(1) >= threshold
        other = None if source == "master" else np.unique(labels)
        full = score(labels, predicted, classes, other)
        result[source] = dict(summarize(full), largest_errors=full["largest_errors"], blight=[
            row for row in full["blight"] if row["total"]
        ], loss=float(-np.log(effective[np.arange(len(labels)), labels].clip(1e-30)).mean()))
        stats[source] = dict(
            switched=switches, suggested=int(selected.sum()),
            correct_suggestions=int(((predicted == labels) & selected).sum()),
            wrong_suggestions=int(((predicted != labels) & selected).sum()),
        )
    master = result.pop("master")
    return {"master": master, "fields": result, "suggestions": stats, "candidate_min": minimum,
            "selection_field_f1": float(np.mean([d["macro_f1"] for k, d in result.items() if k != "banu"]))}


def can_present(case, baseline, rules, plan):
    wrong = case["suggestions"]["master"]["wrong_suggestions"]
    original = baseline["suggestions"]["master"]["wrong_suggestions"]
    rate_delta = (wrong - original) / baseline["master"]["sample_count"]
    return is_eligible(case, baseline, rules) and rate_delta <= plan["max_added_wrong_suggestion_rate"]


def run(release_dir, parent, output, plan_file):
    config = read(parent / "run.json")["config"]
    decision = read(parent / "decision.json")
    if read(parent / "status.json")["state"] != "complete":
        raise ValueError("The matched parent must be complete")
    plan = read(plan_file)
    output.mkdir(parents=True, exist_ok=False)
    frozen, release, previous, new = load_inputs(release_dir, config)
    if release["fingerprint"] != decision["release"]:
        raise ValueError("Parent decision references another field release")
    device = device_for("auto")
    torch.set_num_threads(4)
    predictor = Predictor(ARTIFACT, device)
    if predictor.policy["name"] != "d4_probability_mean":
        raise ValueError("Unexpected reference inference policy")
    spec = predictor.spec
    master = PixelCache("val", spec["input_size"], frozen)
    dev = prepare_development(previous, new, spec["input_size"])

    def master_batches(size):
        for start in range(0, len(master), size):
            yield master.batch(np.arange(start, min(start + size, len(master))))[0]

    def field_batches(raw):
        return raw.split

    data = {"master": (master.labels, master_batches)}
    data.update({name: (labels, field_batches(raw)) for name, (raw, labels) in dev.items()})
    base = collect(predictor, data, config["batch_size"])
    baseline = quantify(base, None, predictor.classes, plan, None)
    out = {"plan": plan, "baseline": baseline, "arms": {}, "release": release["fingerprint"],
           "parent_decision_sha256": sha256(parent / "decision.json")}
    rules = config["selection"]
    preferred = {}
    atomic_json(output / "screening.json", out)
    for arm, entry in decision["arms"].items():
        checkpoint = Path(entry["checkpoint"])
        if sha256(checkpoint) != entry["sha256"]:
            raise ValueError("Parent weights changed")
        weights = torch.load(checkpoint, weights_only=True, map_location="cpu")["weights"]
        predictor.model.load_state_dict(weights, strict=True)
        current = collect(predictor, data, config["batch_size"])
        options = []
        for minimum in plan["field_model_score_thresholds"]:
            choice = quantify(base, current, predictor.classes, plan, minimum)
            choice["eligible"] = can_present(choice, baseline, rules, plan)
            options.append(choice)
            print(arm, minimum, choice["master"]["macro_f1"], choice["selection_field_f1"],
                  choice["suggestions"]["master"], choice["eligible"], flush=True)
            existing = preferred.get(arm)
            if (existing is None or (choice["eligible"] and not existing["eligible"])
                    or choice["eligible"] == existing["eligible"] and choose(choice, existing)):
                preferred[arm] = choice
        out["arms"][arm] = options
        atomic_json(output / "screening.json", out)
    fresh = preferred["new_field"]
    old = preferred["matched_control"]
    passes = fresh["eligible"] and fresh["selection_field_f1"] >= old["selection_field_f1"] + rules["control_min_gain"]
    atomic_json(output / "decision.json", {**out, "selected_options": preferred, "promotion_guards_passed": passes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    run(args.release, args.parent, args.output, args.plan)
