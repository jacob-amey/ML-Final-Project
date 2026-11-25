"""Matched scratch tomato experiment, with frozen 177-label routing checks."""

import argparse
import fcntl
from pathlib import Path
import signal

import numpy as np
import torch

from .common import ROOT, atomic_json, environment, sha256, utc_now
from .data import PixelCache, cache_path
from .field_expansion import load_inputs
from .inference import Predictor
from .potato_policy import distribution, master_probabilities
from .potato_study import matched_training_rows, run_arm, verify_arm
from .scoring import PredictionScores
from .train import device_for
from .upload_study import ARTIFACT, load_study, network, pixels, read, score, summarize


PLAN = ROOT / "configs/leafcheck_tomato_specialist_v1.json"


def data_for(config):
    lineage = read(ROOT / "configs/upload_blight_v1.json")
    frozen, master, first = load_study(lineage)
    pinned = {}
    supplements = {}
    for key, (folder, fingerprint) in config["sources"].items():
        _, release, _, records = load_inputs(ROOT / "data/processed" / folder,
                                             {"initial_weights": lineage["base_checkpoint_sha256"]})
        if release["fingerprint"] != fingerprint:
            raise ValueError(f"Wrong release for {key}")
        pinned[key] = fingerprint
        supplements[key] = records

    classes = set(config["labels"])
    base = [r for r in master if r["class_name"] in classes]
    added = [r for r in first + supplements["field_v2"] + supplements["tom2024"]
             if r["class_name"] in classes and r["split"] == "train"]
    dev = {"master": [r for r in base if r["split"] == "val"]}
    for source in ("sheikh", "banu"):
        dev[source] = [r for r in first if r["class_name"] in classes and r["split"] == "val" and r["source"] == source]
    dev["pakistan"] = supplements["pakistan"]
    training = [r for r in base if r["split"] == "train"]
    combined = training + added + [r for group in dev.values() for r in group]
    if len({r["image_id"] for r in combined}) != len(combined) or len({r["pixel_d4_sha256"] for r in combined}) != len(combined):
        raise ValueError("The study inputs contain duplicate image content")
    train_groups = {r["group_id"] for r in training + added}
    if any(row["group_id"] in train_groups or row["split"] != "val" or row["class_name"] not in classes
           for records in dev.values() for row in records):
        raise ValueError("Development contains training groups or incompatible rows")
    if not all(dev.values()):
        raise ValueError("Every registered source needs development support")
    return frozen, pinned, training, added, dev


def aggregate(per_source, selected):
    keys = ("macro_f1", "accuracy", "loss")
    result = {key: float(np.mean([per_source[source][key] for source in selected])) for key in keys}
    result["confusion_matrix"] = {source: per_source[source]["confusion_matrix"] for source in selected}
    return result


def evaluate(probabilities, truth, classes):
    output = score(truth, probabilities.argmax(1), classes, np.unique(truth))
    output["loss"] = float(-np.log(np.maximum(probabilities[np.arange(len(truth)), truth],
                                               np.finfo(probabilities.dtype).tiny)).mean())
    output["per_class"] = [item for item in output["per_class"] if item["support"]]
    return output


def conditioned(probability, columns):
    scores = probability[:, columns]
    return scores / scores.sum(1, keepdims=True)


def routed(probability, gate, specialist, columns, mix):
    combined = probability.copy()
    block = combined[np.ix_(gate, columns)]
    combined[np.ix_(gate, columns)] = (1 - mix) * block + mix * block.sum(1, keepdims=True) * specialist
    return combined


def score_joint(output, config, frozen, incumbent, columns, originals, dev, decision, spec, run):
    device = incumbent.device
    cache = PixelCache("val", incumbent.spec["input_size"], frozen)
    master_probability = master_probabilities(incumbent, cache)
    full_inputs = {**originals, "master_full": (master_probability, cache.labels)}
    gates = {name: np.flatnonzero(np.isin(values.argmax(1), columns))
             for name, (values, _) in full_inputs.items()}
    raw_images = {name: dev[name][gate] for name, gate in gates.items() if name != "master_full"}
    registry = ROOT / "data/processed/leafcheck177_final"
    master_gate = [dict(cache.rows[i], path=registry / cache.rows[i]["output_relative_path"])
                   for i in gates["master_full"]]
    raw_images["master_full"] = pixels(master_gate, spec["input_size"])
    baseline = {name: summarize(evaluate(values, y, incumbent.classes))
                for name, (values, y) in full_inputs.items()}
    sources = config["selection_sources"]
    master = baseline["master_full"]
    rules = config["selection"]
    effect = {}
    for arm, results in decision["arms"].items():
        effect[arm] = []
        for entry in results:
            checkpoint = Path(entry["checkpoint"])
            if sha256(checkpoint) != entry["sha256"]:
                raise ValueError("The selected checkpoint changed before evaluation")
            saved = verify_arm(checkpoint, config, run)
            leaf = network(config, device)
            leaf.load_state_dict(saved["weights"], strict=True)
            scorer = PredictionScores(leaf, {"name": "d4_probability_mean"}).eval()
            scores = {}
            for name, (p, y) in full_inputs.items():
                gate = gates[name]
                if len(gate):
                    r = distribution(scorer, raw_images[name], spec, device, size=config["batch_size"])
                    p = routed(p, gate, r, columns, config["joint_blend"])
                scores[name] = summarize(evaluate(p, y, incumbent.classes))
            row = scores["master_full"]
            gain = float(np.mean([scores[name]["macro_f1"] - baseline[name]["macro_f1"] for name in sources]))
            guard = all(row[key] >= master[key] - rules["max_joint_master_drop"] for key in ("macro_f1", "accuracy"))
            effect[arm].append({"seed": entry["seed"], "scores": scores, "joint_field_gain": gain,
                                "passed_master_guard": guard})
    field = effect["field"]
    control = effect["control"]
    passed = (decision["conditional_eligible"]
              and all(row["passed_master_guard"] and row["joint_field_gain"] >= rules["joint_field_min_gain"]
                      for row in field)
              and all(candidate["joint_field_gain"] > ref["joint_field_gain"] for candidate, ref in zip(field, control)))
    record = {"baseline": baseline, "matched_arms": effect, "gate_counts": {k: len(v) for k, v in gates.items()},
              "research_confirmation_eligible": passed, "test_opened": False, "evaluator_sha256": sha256(Path(__file__))}
    atomic_json(output / "joint.json", record)
    print("Joint routing", effect, "eligible", passed, flush=True)


def run_study(output, config, resume, joint_only):
    if (output / "joint.json").exists():
        print("Preserving completed study.", flush=True)
        return
    torch.set_num_threads(4)
    device = device_for("auto")
    frozen, releases, master, extra, dev = data_for(config)
    classes = config["labels"]
    spec = read(cache_path(config["input_size"], frozen["fingerprint"]) / "preprocessing.json")
    incumbent = Predictor(ARTIFACT, device)
    if classes != [label for label in incumbent.classes if label.startswith("tomato__")]:
        raise ValueError("Study must preserve the complete ordered tomato scope")
    mapping = {label: i for i, label in enumerate(classes)}
    columns = np.array([incumbent.classes.index(label) for label in classes])
    selected = config["selection_sources"]
    run = {"config": config, "releases": releases, "master": frozen["fingerprint"], "spec": spec,
           "reference_checkpoint": incumbent.metadata["checkpoint_sha256"], "trainer_sha256": sha256(Path(__file__)),
           "helper_sha256": sha256(Path(__file__).with_name("potato_study.py"))}
    receipt = output / "run.json"
    decision_path = output / "conditional.json"
    if joint_only:
        if not receipt.exists() or not decision_path.exists():
            raise ValueError("Joint-only scoring requires the completed matched training decision")
        trained_run = read(receipt)["run"]
        if any(trained_run[key] != run[key] for key in run if key != "trainer_sha256"):
            raise ValueError("Joint-only inputs differ from the completed training run")
        run = trained_run
    elif receipt.exists():
        if not resume or read(receipt)["run"] != run:
            raise ValueError("Use --resume with the unchanged original inputs and recipe")
    else:
        atomic_json(receipt, {"run": run, "environment": environment(device), "started_utc": utc_now()})

    def sample(records, size):
        return pixels(records, size), np.array([mapping[r["class_name"]] for r in records])

    val = {}
    original = {}
    for name, records in dev.items():
        val[name], _ = sample(records, spec["input_size"])
        raw, labels = sample(records, incumbent.spec["input_size"])
        p = distribution(incumbent.scoring_model, raw, incumbent.spec, device, size=config["batch_size"])
        original[name] = p, labels
    incumbent_scores = {name: evaluate(conditioned(p, columns), labels, classes)
                        for name, (p, labels) in original.items()}
    targets = {name: np.array([mapping[row["class_name"]] for row in group]) for name, group in dev.items()}
    baseline = {"master": incumbent_scores["master"], "field": aggregate(incumbent_scores, selected),
                "by_source": incumbent_scores}

    def assess(scorer):
        scores = {name: evaluate(distribution(scorer, raw, spec, device, size=config["batch_size"]), targets[name], classes)
                  for name, raw in val.items()}
        return {"master": scores["master"], "field": aggregate(scores, selected), "by_source": scores}

    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if decision_path.exists():
        decision = read(decision_path)
    else:
        data, labels = sample(master + extra, spec["input_size"])
        tensor_labels = torch.from_numpy(labels)
        positions = {row["image_id"]: i for i, row in enumerate(master + extra)}
        arms = {"control": [], "field": []}
        print("Train originals", len(master), len(extra), "development", {k: len(v) for k, v in dev.items()},
              "baseline", baseline["field"]["macro_f1"], flush=True)
        for seed in config["seeds"]:
            for arm in arms:
                if stop["requested"]:
                    return
                rows = matched_training_rows(master, extra, classes, seed) if arm == "control" else master + extra
                indices = [positions[r["image_id"]] for r in rows]
                train = (data[indices], tensor_labels[indices])
                result = run_arm(output / f"{arm}-{seed}", config, arm, seed, train, val, baseline, spec, device, stop,
                                 run, assess=assess)
                if result is None:
                    return
                arms[arm].append(result)
        control, field = ([row["scores"]["field"]["macro_f1"] for row in arms[key]] for key in ("control", "field"))
        rules = config["selection"]
        eligible = (all(row["passed_master_limits"] for row in arms["field"])
                    and all(a > b for a, b in zip(field, control))
                    and np.mean(field) >= np.mean(control) + rules["field_min_gain_average_vs_control"]
                    and np.mean(field) >= baseline["field"]["macro_f1"] + rules["field_min_gain_average_vs_masked"])
        decision = {"conditional_eligible": bool(eligible), "arms": arms, "reference": baseline,
                    "training": {"master": len(master), "field": len(extra)}, "completed_utc": utc_now()}
        atomic_json(decision_path, decision)
        print("Conditional field", field, "control", control, "eligible", eligible, flush=True)
    joint_inputs = {name: (p, np.array([columns[label] for label in labels]))
                    for name, (p, labels) in original.items()}
    if not stop["requested"]:
        score_joint(output, config, frozen, incumbent, columns, joint_inputs, val, decision, spec, run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PLAN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--joint-only", action="store_true", help="Score an already completed matched run without training")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_study(args.output, read(args.config), args.resume, args.joint_only)
