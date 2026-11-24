"""Try a guarded seven-way specialist with group-sampled other-label negatives."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .common import ROOT, atomic_checkpoint, atomic_json, environment, seed_everything, sha256, write_csv
from .data import cache_path
from .train import cpu_state, device_for, restore_rng, rng_state
from .upload_study import (SAVED_VALIDATION, load_study, network, pixels, probabilities,
                           read, rows, run_epoch, score, summarize, training_set)


def negative_rows(master, focus, config):
    by_label = defaultdict(lambda: defaultdict(list))
    for row in master:
        if row["split"] == "train" and row["class_name"] not in focus:
            by_label[row["class_name"]][row["group_id"]].append(row)
    rng = np.random.default_rng(config["negative_sampling_seed"])
    result = []
    for label in sorted(by_label):
        groups = sorted(by_label[label])
        rng.shuffle(groups)
        for group in groups[:config["negative_groups_per_class"]]:
            group_rows = sorted(by_label[label][group], key=lambda r: r["image_id"])
            result.append(group_rows[int(rng.integers(len(group_rows)))])
    return result


def gate_predictions(base, confidence, p, focus, threshold, maximum):
    chosen = p.argmax(1)
    active = (chosen < len(focus)) & (p.max(1) >= threshold) & (confidence <= maximum)
    result = base.copy()
    result[active] = focus[chosen[active]]
    return result, active


def evaluate(base, confidence, p, focus, threshold, config, targets, classes, splits, details=False):
    proposed, active = gate_predictions(base, confidence, p, focus, threshold, config["maximum_incumbent_score"])
    result = {}
    for name, keep in splits.items():
        y = targets[keep]
        result[name] = score(y, proposed[keep], classes, None if name == "master" else np.unique(y))
        result[name]["routed"] = int(active[keep].sum())
        result[name]["changed"] = int(np.count_nonzero(proposed[keep] != base[keep]))
        if not details:
            result[name] = {**summarize(result[name]), "routed": result[name]["routed"],
                            "changed": result[name]["changed"]}
    return result, proposed


def better_rule(a, b):
    if b is None:
        return True
    x, y = a["master"]["macro_f1"], b["master"]["macro_f1"]
    return x > y + 1e-6 or (abs(x - y) <= 1e-6 and a["supplement"]["macro_f1"] > b["supplement"]["macro_f1"] + 1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/upload_blight_router_v1.json")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=434)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--master-only", action="store_true", help="Matched control without the screened supplement")
    args = parser.parse_args()
    config = read(args.config)
    base_file = ROOT / config["base_config"]
    train = read(base_file)
    frozen, master, supplement = load_study(train)
    classes = frozen["classes"]
    named = train["labels"]
    focus = np.array([classes.index(name) for name in named])
    run = args.input / f"router_{'master' if args.master_only else 'expanded'}_seed{args.seed}"
    if not args.resume:
        run.mkdir(exist_ok=False)
    elif read(run / "status.json")["state"] == "complete":
        raise ValueError("This run is complete")
    val = sorted((r for r in master if r["split"] == "val"), key=lambda r: r["image_id"])
    extra = sorted((r for r in supplement if r["split"] == "val"), key=lambda r: r["image_id"])
    base_val = {r["image_id"]: r for r in rows(SAVED_VALIDATION)}
    base_extra = {r["image_id"]: r for r in rows(args.input / "incumbent_scored_rows.csv")
                  if r["source"] in {"sheikh", "banu", "bugwood"}}
    predictions = [base_val[r["image_id"]] for r in val] + [base_extra[r["image_id"]] for r in extra]
    assert len(base_val) == len(val) and len(base_extra) == len(extra)
    for r, pred in zip(val + extra, predictions):
        assert r["class_name"] == pred.get("actual_class", pred.get("true"))
    base = np.array([classes.index(r.get("predicted_class", r.get("predicted"))) for r in predictions])
    confidence = np.array([float(r["score"]) for r in predictions])
    targets = np.array([int(r["class_id"]) for r in val + extra])
    splits = {"master": np.arange(len(val)), "supplement": np.array([
        len(val) + i for i, r in enumerate(extra) if r["source_slice"] != "bugwood"])}
    negatives = negative_rows(master, set(named), config)
    fit = training_set(master, supplement, train, {"supplement": not args.master_only}) + negatives
    train = {**train, "labels": named + ["other"]}
    indices = {name: idx for idx, name in enumerate(named)}
    y = torch.tensor([indices.get(r["class_name"], len(named)) for r in fit])
    torch.set_num_threads(4)
    device = device_for(args.device)
    spec = read(cache_path(train["input_size"], frozen["fingerprint"]) / "preprocessing.json")
    fit_pixels = pixels(fit, train["input_size"])
    eval_pixels = pixels(val + extra, train["input_size"])
    seed_everything(args.seed)
    model = network(train, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train["learning_rate"], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    weights = torch.ones(len(named) + 1, device=device)
    identity = {"config_sha256": sha256(args.config), "base_config_sha256": sha256(base_file),
                "dataset_fingerprint": frozen["fingerprint"], "supplement_fingerprint": train["supplement_fingerprint"],
                "training_membership_sha256": hashlib.sha256("\n".join(r["image_id"] for r in fit).encode()).hexdigest(),
                "n_other": len(negatives), "n_target": len(fit) - len(negatives),
                "seed": args.seed, "device": str(device), "preprocessing": spec, "labels": train["labels"]}
    incumbent, _ = evaluate(base, confidence, np.zeros((len(targets), len(named) + 1)), focus,
                            1.0, config, targets, classes, splits)
    best, best_weights, history = None, None, []
    if args.resume:
        state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        assert state["identity"] == identity
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        best, best_weights, history = state["best"], state["best_weights"], state["history"]
    else:
        write_csv(run / "train_manifest.csv", [
            {"image_id": r["image_id"], "path": str(r["path"].relative_to(ROOT)), "class": r["class_name"],
             "group": r["group_id"], "source": r["source_slice"], "specialist_index": int(index), "sha256": r["sha256"]}
            for r, index in zip(fit, y)])
        atomic_json(run / "configuration.json", config)
        atomic_json(run / "environment.json", environment(device))
    best_eligible, eligible_weights = None, None
    if args.resume:
        best_eligible, eligible_weights = state["best_eligible"], state["eligible_weights"]
    for epoch in range(len(history) + 1, config["max_epochs"] + 1):
        began = time.perf_counter()
        loss = run_epoch(model, optimizer, fit_pixels, y, spec, device, train, epoch, args.seed, weights)
        p = probabilities(model, eval_pixels, spec, device)
        epoch_results = []
        for threshold in config["minimum_specialist_score"]:
            result, _ = evaluate(base, confidence, p, focus, threshold, config, targets, classes, splits)
            result.update(epoch=epoch, threshold=threshold, loss=loss)
            m, s = result["master"], result["supplement"]
            m0, s0 = incumbent["master"], incumbent["supplement"]
            is_eligible = (m["macro_f1"] - m0["macro_f1"] >= config["minimum_master_macro_f1_gain"]
                           and s["macro_f1"] - s0["macro_f1"] >= config["minimum_supplement_macro_f1_gain"]
                           and m["accuracy"] >= m0["accuracy"] and m["wrong_crop"] <= m0["wrong_crop"])
            result["eligible"] = is_eligible
            epoch_results.append(result)
            if better_rule(result, best):
                best, best_weights = result, cpu_state(model)
            if is_eligible and better_rule(result, best_eligible):
                best_eligible, eligible_weights = result, cpu_state(model)
        scheduler.step(max(r["master"]["macro_f1"] for r in epoch_results))
        history.append({"epoch": epoch, "training_loss": loss, "seconds": time.perf_counter() - began,
                        "candidates": epoch_results})
        atomic_checkpoint(run / "last.pt", {"identity": identity, "weights": cpu_state(model),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "rng": rng_state(), "history": history, "best": best, "best_weights": best_weights,
                          "best_eligible": best_eligible, "eligible_weights": eligible_weights})
        atomic_json(run / "history.json", history)
        atomic_json(run / "status.json", {"state": "running", "best": best,
                    "best_eligible": best_eligible, "completed_epochs": epoch})
        print(json.dumps(history[-1]), flush=True)
        latest = max(best["epoch"], best_eligible["epoch"] if best_eligible else 0)
        if epoch >= config["min_epochs"] and epoch - latest >= config["patience"]:
            break
    selected, saved_weights = (best_eligible, eligible_weights) if best_eligible else (best, best_weights)
    checkpoint = run / "best_model.pt"
    atomic_checkpoint(checkpoint, {"identity": identity, "weights": saved_weights, "selection": selected})
    restored = network(train, device)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["weights"])
    p = probabilities(restored, eval_pixels, spec, device)
    splits["field_diagnostic"] = np.array([len(val) + i for i, r in enumerate(extra) if r["source_slice"] == "bugwood"])
    results, proposed = evaluate(base, confidence, p, focus, selected["threshold"], config, targets, classes, splits, True)
    assert abs(results["master"]["macro_f1"] - selected["master"]["macro_f1"]) < 1e-12
    assert abs(results["supplement"]["macro_f1"] - selected["supplement"]["macro_f1"]) < 1e-12
    write_csv(run / "predictions.csv", [
        {"image_id": r["image_id"], "source": r["source_slice"], "true": r["class_name"], "group_id": r["group_id"],
         "before": classes[old], "after": classes[new], "specialist_score": float(probs.max())}
        for r, old, new, probs in zip(val + extra, base, proposed, p)])
    atomic_json(run / "validation_metrics.json", results)
    atomic_json(run / "status.json", {"state": "complete", "best": best, "best_eligible": best_eligible,
                "checkpoint_sha256": sha256(checkpoint), "reload_verified": True,
                "completed_epochs": len(history), "identity": identity})
    print("COMPLETE", json.dumps(selected), flush=True)


if __name__ == "__main__":
    main()
