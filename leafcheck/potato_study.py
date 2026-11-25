"""Train a custom known-potato CNN in matched arms without opening any test images."""

import argparse
import fcntl
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn

from .common import ROOT, atomic_checkpoint, atomic_json, better, environment, metrics_from_matrix, seed_everything, sha256, utc_now
from .data import cache_path
from .field_expansion import load_inputs
from .inference import Predictor
from .preprocessing import prepare_batch
from .scoring import PredictionScores
from .train import cpu_state, device_for, restore_rng, rng_state
from .upload_study import ARTIFACT, load_study, network, pixels, read, run_epoch

V2 = ROOT / "data/processed/leafcheck_field_v2_sampled"
PLAN = ROOT / "configs/leafcheck_potato_v1.json"
FIELD_CONFIG = ROOT / "configs/leafcheck_field_v2_full.json"


def matched_training_rows(master, extra, labels, seed):
    rng = np.random.default_rng(seed)
    pools = {name: [r for r in master if r["class_name"] == name] for name in labels}
    if any(not rows for rows in pools.values()):
        raise ValueError("Missing master training label")
    return master + [pools[r["class_name"]][rng.integers(len(pools[r["class_name"]]))] for r in extra]


@torch.inference_mode()
def score_pixels(scorer, samples, spec, classes, device, batch_size, columns=None):
    scorer.eval()
    raw, labels = samples
    values = []
    for batch in raw.split(batch_size):
        values.append(scorer(prepare_batch(batch, spec, device)).cpu())
    logits = torch.cat(values)
    if columns is not None:
        logits = logits[:, columns]
    loss = nn.functional.cross_entropy(logits, labels).item()
    predictions = logits.argmax(1).numpy()
    targets = labels.numpy()
    n = len(classes)
    matrix = np.bincount(n * targets + predictions, minlength=n * n).reshape(n, n)
    return metrics_from_matrix(matrix, classes, loss)


def tied_metrics(scores, config, baseline):
    rules = config["selection"]
    main, ref = scores["master"], baseline["master"]
    return (main["macro_f1"] >= ref["macro_f1"] - rules["master_max_macro_f1_drop_per_seed"]
            and main["accuracy"] >= ref["accuracy"] - rules["master_max_accuracy_drop_per_seed"])


def verify_arm(path, config, run):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != config or state["run"] != run:
        raise ValueError("Saved training state belongs to another study")
    return state


def run_arm(directory, config, arm, seed, train, val, baseline, spec, device, stop, run, assess=None):
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    model = network(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    checkpoint = directory / "last.pt"
    history, best, eligible = [], None, None
    if checkpoint.exists():
        state = verify_arm(checkpoint, config, run)
        model.load_state_dict(state["weights"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        history, best, eligible = state["history"], state["best"], state["eligible"]
    ordered = config["labels"]
    raw, labels = train
    counts = labels.bincount(minlength=len(ordered)).float()
    weights = counts.rsqrt()
    weights = weights / ((counts * weights).sum() / counts.sum())
    weights = weights.to(device)
    scorer = PredictionScores(model, {"name": "d4_probability_mean"}).eval()
    if assess is None:
        def assess(scorer):
            return {name: score_pixels(scorer, sample, spec, ordered, device, config["batch_size"])
                    for name, sample in val.items()}

    for epoch in range(len(history) + 1, config["max_epochs"] + 1):
        chosen = eligible or best
        if epoch > config["min_epochs"] and chosen and epoch - chosen["epoch"] > config["patience"]:
            break
        started = time.monotonic()
        loss = run_epoch(model, optimizer, raw, labels, spec, device, config, epoch, seed, weights)
        measures = assess(scorer)
        result = {"epoch": epoch, "scores": measures, "train_loss": loss, "seconds": time.monotonic() - started}
        history.append(result)
        prev_best = best["scores"]["field"] if best else None
        improved = better(measures["field"], prev_best)
        if improved:
            best = result
        prev_eligible = eligible["scores"]["field"] if eligible else None
        improved_eligible = tied_metrics(measures, config, baseline) and better(measures["field"], prev_eligible)
        weights_cpu = cpu_state(model)
        if improved_eligible:
            eligible = result
        if improved or improved_eligible:
            atomic_checkpoint(directory / f"epoch-{epoch:03d}.pt", {
                "weights": weights_cpu, "config": config, "run": run, "result": result,
            })
        atomic_checkpoint(checkpoint, {"weights": weights_cpu, "optimizer": optimizer.state_dict(), "rng": rng_state(),
                                      "history": history, "best": best, "eligible": eligible, "config": config, "run": run})
        atomic_json(directory / "history.json", history)
        print(arm, seed, epoch, "field", measures["field"]["macro_f1"], "master", measures["master"]["macro_f1"], flush=True)
        if stop["requested"]:
            return None
    result = eligible or best
    model_path = directory / f"epoch-{result['epoch']:03d}.pt"
    saved = verify_arm(model_path, config, run)
    model.load_state_dict(saved["weights"], strict=True)
    replay = assess(scorer)["field"]
    assert replay["confusion_matrix"] == result["scores"]["field"]["confusion_matrix"]
    return {**result, "checkpoint": str(model_path), "sha256": sha256(model_path), "completed_epochs": len(history),
            "passed_master_limits": bool(eligible), "seed": seed, "checkpoint_reload_verified": True}


def train(output, config, resume):
    if (output / "decision.json").exists():
        print("Preserving the completed run", flush=True)
        return
    device = device_for("auto")
    torch.set_num_threads(4)
    positives = set(config["labels"])
    lineage = read(FIELD_CONFIG)
    frozen, release, old, new = load_inputs(V2, lineage)
    _, master, _ = load_study(read(ROOT / "configs/upload_blight_v1.json"))
    spec = read(cache_path(config["input_size"], frozen["fingerprint"]) / "preprocessing.json")
    classes = config["labels"]
    run = {"master_fingerprint": frozen["fingerprint"], "field_fingerprint": release["fingerprint"],
           "base_checkpoint": lineage["initial_weights"], "spec": spec, "config": config,
           "trainer_sha256": sha256(Path(__file__))}
    run_path = output / "run.json"
    if run_path.exists():
        if not resume or read(run_path)["run"] != run:
            raise ValueError("Use --resume with the unchanged registered run")
    else:
        atomic_json(run_path, {"run": run, "environment": environment(device), "started_utc": utc_now()})
    base = [r for r in master if r["split"] == "train" and r["class_name"] in positives]
    extra = [r for r in old + new if r["split"] == "train" and r["class_name"] in positives]
    dev = {
        "master": [r for r in master if r["split"] == "val" and r["class_name"] in positives],
        "field": [r for r in new if r["split"] == "val" and r["source"] == "mduma" and r["class_name"] in positives],
    }
    all_training = {r["image_id"] for r in base + extra}
    assert len(all_training) == len(base) + len(extra) and all(
        not all_training.intersection(r["image_id"] for r in records) for records in dev.values())

    def sample(records, size):
        return pixels(records, size), torch.tensor([classes.index(r["class_name"]) for r in records], dtype=torch.long)

    original = Predictor(ARTIFACT, device)
    columns = [original.classes.index(name) for name in classes]
    baseline = {name: score_pixels(original.scoring_model, sample(records, original.spec["input_size"]),
                                   original.spec, classes, device, config["batch_size"], columns)
                for name, records in dev.items()}
    val = {name: sample(records, config["input_size"]) for name, records in dev.items()}
    base_pixels = {r["image_id"]: i for i, r in enumerate(base + extra)}
    training = sample(base + extra, config["input_size"])
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    results = {"control": [], "field": []}
    print("Training unique", len(base), len(extra), "and validation",
          {k: len(v) for k, v in dev.items()}, "masked field", baseline["field"]["macro_f1"], flush=True)
    for seed in config["seeds"]:
        for arm in results:
            if stop["requested"]:
                return
            matches = matched_training_rows(base, extra, classes, seed) if arm == "control" else base + extra
            indices = [base_pixels[r["image_id"]] for r in matches]
            result = run_arm(output / f"{arm}-{seed}", config, arm, seed, tuple(x[indices] for x in training),
                             val, baseline, spec, device, stop, run)
            if result is None:
                return
            results[arm].append(result)

    def average(arm):
        return float(np.mean([r["scores"]["field"]["macro_f1"] for r in results[arm]]))

    rules = config["selection"]
    improvements = [candidate["scores"]["field"]["macro_f1"] - control["scores"]["field"]["macro_f1"]
                    for candidate, control in zip(results["field"], results["control"])]
    passed = (all(r["passed_master_limits"] for r in results["field"]) and min(improvements) > 0
              and average("field") >= average("control") + rules["field_min_gain_average_vs_control"]
              and average("field") >= baseline["field"]["macro_f1"] + rules["field_min_gain_average_vs_masked"])
    chosen = max(results["field"], key=lambda r: r["scores"]["field"]["macro_f1"])
    atomic_json(output / "decision.json", {"eligible": passed, "run": run, "arms": results,
                                            "diagnostic_candidate": chosen, "masked_incumbent": baseline,
                                            "per_seed_field_gains": improvements,
                                            "training_counts": {"unique_master": len(base), "unique_field": len(extra)},
                                            "completed_utc": utc_now()})
    print("Field macro-F1", average("field"), "control", average("control"), "passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PLAN)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args.output, read(args.config), args.resume)
