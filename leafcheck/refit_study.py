"""Compare linear-head refinements without opening either established holdout."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .common import ROOT, atomic_checkpoint, atomic_json, environment, seed_everything, sha256, write_csv
from .data import PixelCache
from .inference import Predictor
from .preprocessing import prepare_batch
from .train import cpu_state, device_for
from .upload_study import ARTIFACT, SAVED_VALIDATION, load_study, pixels, read, rows, score, summarize

CONFIG = ROOT / "configs/leafcheck_refit_v1.json"


def save_np(path, value):
    stage = path.with_suffix(".tmp.npy")
    np.save(stage, value)
    stage.replace(path)


@torch.inference_mode()
def extract(model, raw, spec, device, views):
    x = prepare_batch(raw, spec, device)
    if views == 1:
        return model.classifier[:-1](model.features(x)).cpu().numpy()[None]
    return np.stack([
        model.classifier[:-1](model.features(torch.rot90(v, n, (-2, -1))
            .contiguous(memory_format=torch.channels_last))).cpu().numpy()
        for v in (x, x.flip(-1)) for n in range(4)
    ])


def make_features(args, cfg, frozen, master, extra):
    predictor = Predictor(ARTIFACT, device_for(args.device))
    assert predictor.metadata["checkpoint_sha256"] == cfg["base_checkpoint"]
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for split in ("train", "val"):
        cache = PixelCache(split, 224, frozen)
        membership = {r["image_id"]: r for r in master if r["split"] == split}
        from_master = [membership[r["image_id"]] for r in cache.rows]
        from_extra = [r for r in extra if r["split"] == split]
        views = 1 if split == "train" else 8
        width = predictor.model.classifier[-1].in_features
        matrix = np.empty((views, len(from_master) + len(from_extra), width), np.float32)
        for start in range(0, len(cache), 128):
            end = min(start + 128, len(cache))
            raw, _ = cache.batch(np.arange(start, end))
            matrix[:, start:end] = extract(predictor.model, raw, predictor.spec, predictor.device, views)
            if start % 8192 == 0:
                print(split, f"{end}/{len(cache)}", flush=True)
        more = pixels(from_extra, 224)
        for start in range(0, len(more), 128):
            end = min(start + 128, len(more))
            matrix[:, len(cache)+start:len(cache)+end] = extract(
                predictor.model, more[start:end], predictor.spec, predictor.device, views)
        where = args.output / (split + ".npy")
        save_np(where, matrix)
        write_csv(args.output / (split + ".csv"), [{
            "image_id": r["image_id"], "group_id": r["group_id"], "class_id": r["class_id"],
            "class_name": r["class_name"], "source": r["source_slice"],
            "master": i < len(from_master), "path": str(r["path"].relative_to(ROOT))
        } for i, r in enumerate(from_master + from_extra)])
        receipts[split] = {"features_sha256": sha256(where), "shape": list(matrix.shape),
                           "membership_sha256": sha256(args.output / (split + ".csv"))}
    atomic_json(args.output / "features.json", {
        "splits": receipts, "base_checkpoint": cfg["base_checkpoint"],
        "frozen_fingerprint": frozen["fingerprint"], "device": str(predictor.device),
        "supplement_fingerprint": read(ROOT / cfg["base_config"])["supplement_fingerprint"]})


def load_features(output, cfg, frozen):
    lock = read(output / "features.json")
    assert lock["base_checkpoint"] == cfg["base_checkpoint"] and lock["frozen_fingerprint"] == frozen["fingerprint"]
    result = {}
    for split in ("train", "val"):
        stamp = lock["splits"][split]
        table = output / f"{split}.csv"
        features = output / f"{split}.npy"
        assert sha256(table) == stamp["membership_sha256"] and sha256(features) == stamp["features_sha256"]
        result[split] = torch.from_numpy(np.load(features)), rows(table)
    return result


def split_scores(p, members, classes):
    predictions = p.argmax(1)
    y = np.array([int(row["class_id"]) for row in members])
    master = np.array([r["master"] == "True" for r in members])
    origins = np.array([r["source"] for r in members])
    sets = {"master": master, "supplement": ~master & (origins != "bugwood"),
            "field_diagnostic": origins == "bugwood"}
    return {name: score(y[mask], predictions[mask], classes,
                        None if name == "master" else np.unique(y[mask]))
            for name, mask in sets.items()}


@torch.inference_mode()
def predict_head(head, features, device):
    return np.concatenate([
        head(chunk.to(device)).softmax(2).mean(0).cpu().numpy()
        for chunk in features.split(1024, dim=1)
    ])


def fit(args, cfg, frozen):
    device = torch.device("cpu")
    torch.set_num_threads(4)
    feature_sets = load_features(args.output, cfg, frozen)
    train, tr = feature_sets["train"]
    val, va = feature_sets["val"]
    classes = frozen["classes"]
    selected = np.array([r["master"] == "True" for r in va])
    reference = Predictor(ARTIFACT)
    original = reference.model.classifier[-1]
    baseline_p = predict_head(original, val, device)
    saved = {r["image_id"]: int(r["predicted_index"]) for r in rows(SAVED_VALIDATION)}
    expected = [saved[r["image_id"]] for r in va if r["master"] == "True"]
    assert np.array_equal(baseline_p[selected].argmax(1), expected), "Frozen scoring parity failed"
    np.save(args.output / "incumbent_validation_probabilities.npy", baseline_p)
    baseline = split_scores(baseline_p, va, classes)
    atomic_json(args.output / "baseline.json", baseline)
    print("INCUMBENT", summarize(baseline["master"]), summarize(baseline["supplement"]), flush=True)
    if not args.recipe:
        return
    seed_everything(args.seed)
    config = cfg["recipes"][args.recipe]
    run = args.output / f"{args.recipe}_seed{args.seed}"
    if args.resume:
        if read(run / "status.json")["state"] == "complete":
            raise ValueError("This head study run is complete")
    else:
        run.mkdir(exist_ok=False)
    index = np.flatnonzero([r["master"] == "True" or config["added"] for r in tr])
    targets = torch.tensor([int(tr[i]["class_id"]) for i in index])
    train_x = train[0, index]
    groups = Counter(tr[i]["group_id"] for i in index)
    weights = torch.tensor([groups[tr[i]["group_id"]] ** -config["group_power"] for i in index])
    weights /= weights.mean()
    head = nn.Linear(original.in_features, len(classes))
    head.load_state_dict(original.state_dict())
    origin = {k: param.detach().clone() for k, param in original.state_dict().items()}
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=.0001)
    set_crop = sorted({label.split("__", 1)[0] for label in classes})
    crop_id = torch.tensor([set_crop.index(label.split("__", 1)[0]) for label in classes])
    crop_map = nn.functional.one_hot(crop_id, len(set_crop)).float()
    history = []
    best = None
    max_epochs = cfg["epoch_budget"]
    identity = {"config_sha256": sha256(args.config), "features_sha256": sha256(args.output / "features.json"),
                "data": frozen["fingerprint"], "seed": args.seed, "recipe": args.recipe,
                "training_count": len(index), "base": cfg["base_checkpoint"]}
    if args.resume:
        state = torch.load(run / "last.pt", weights_only=True, map_location="cpu")
        assert state["identity"] == identity
        head.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        history, best = state["history"], state["best"]
        assert len(history) <= max_epochs
    else:
        atomic_json(run / "environment.json", environment(device))
    # The balanced-softmax term applies only to the training loss.
    prior = torch.bincount(targets, minlength=len(classes)).float().log() * config.get("prior_power", 0.)
    epoch = len(history)
    if best and epoch >= 7 and epoch - best["epoch"] >= cfg["patience"]:
        max_epochs = epoch
    for epoch in range(epoch + 1, max_epochs + 1):
        start = time.perf_counter()
        order = torch.randperm(len(targets))
        train_loss = 0.0
        for batch in order.split(cfg["batch_size"]):
            logits = head(train_x[batch])
            loss = (nn.functional.cross_entropy(logits + prior, targets[batch], reduction="none") * weights[batch]).mean()
            if config["crop_loss"]:
                crop_p = (logits.softmax(1) @ crop_map).clamp_min(1e-12)
                loss += config["crop_loss"] * nn.functional.nll_loss(crop_p.log(), crop_id[targets[batch]])
            loss += cfg["l2_to_original"] * sum((p - origin[k]).square().mean() for k, p in head.named_parameters())
            assert torch.isfinite(loss).item()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch)
        p = predict_head(head, val, device)
        metric = split_scores(p, va, classes)
        master = summarize(metric["master"])
        second = summarize(metric["supplement"])
        headline = {"epoch": epoch, "master": master, "supplement": second,
                    "seconds": time.perf_counter() - start, "train_loss": train_loss / len(index)}
        history.append(headline)
        if best is None or master["macro_f1"] > best["master"]["macro_f1"] + 1e-6:
            best = headline
            atomic_checkpoint(run / "head.pt", {"state_dict": cpu_state(head), "identity": identity, "validation": best})
        atomic_checkpoint(run / "last.pt", {"identity": identity, "model_state_dict": cpu_state(head),
                          "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                          "history": history, "best": best})
        atomic_json(run / "history.json", history)
        atomic_json(run / "status.json", {"state": "running", "best": best, "completed_epochs": epoch})
        print(args.recipe, json.dumps(headline), flush=True)
        if epoch >= 7 and epoch - best["epoch"] >= cfg["patience"]:
            break
    head.load_state_dict(torch.load(run / "head.pt", weights_only=True)["state_dict"])
    candidate_p = predict_head(head, val, device)
    metrics = split_scores(candidate_p, va, classes)
    assert summarize(metrics["master"]) == best["master"]
    atomic_json(run / "validation.json", metrics)
    np.save(run / "validation_probabilities.npy", candidate_p)
    atomic_json(run / "status.json", {"state": "complete", "best": best, "completed_epochs": epoch,
                "sha256": sha256(run / "head.pt"), "identity": identity, "reload_verified": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["cache", "fit"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--recipe")
    parser.add_argument("--seed", type=int, default=434)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = read(args.config)
    if args.recipe and args.recipe not in cfg["recipes"]:
        parser.error("Unknown recipe in the provided configuration")
    if args.resume and (args.action != "fit" or not args.recipe):
        parser.error("--resume requires fit and --recipe")
    if args.action == "cache":
        if (args.output / "features.json").exists():
            raise FileExistsError("Existing feature cache")
        frozen, master, extra = load_study(read(ROOT / cfg["base_config"]))
        make_features(args, cfg, frozen, master, extra)
    else:
        frozen, _, _ = load_study(read(ROOT / cfg["base_config"]))
        fit(args, cfg, frozen)


if __name__ == "__main__":
    main()
