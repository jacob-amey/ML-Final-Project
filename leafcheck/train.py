"""Development training: this module only opens the training and validation splits."""

import argparse
import json
import math
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
import torch
from torch import nn
from .common import (ROOT, RUNS, atomic_checkpoint, atomic_json, better, environment,
                     metrics_from_matrix, seed_everything, update_matrix, utc_now, write_csv)
from .data import PixelCache, cache_path, release
from .model import LeafCNN
from .preprocessing import prepare_batch


def device_for(name):
    if name == "auto":
        name = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def cpu_state(model):
    return {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
            "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}


def restore_rng(state):
    random.setstate(state["python"]); torch.set_rng_state(state["torch"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    if state["mps"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def crop_marginal_loss(logits, targets, same_crop):
    """Negative log probability of the target crop under the 177-label distribution."""
    same_crop_logits = logits.masked_fill(~same_crop[targets], -torch.inf)
    return torch.logsumexp(logits, dim=1) - torch.logsumexp(same_crop_logits, dim=1)


@torch.inference_mode()
def evaluate(model, cache, spec, device, classes, batch_size=256, keep_predictions=False):
    model.eval()
    matrix = np.zeros((len(classes), len(classes)), np.int64)
    loss_sum = 0.0; predictions = []
    for start in range(0, len(cache), batch_size):
        indices = np.arange(start, min(start + batch_size, len(cache)))
        raw, labels = cache.batch(indices)
        logits = model(prepare_batch(raw, spec, device))
        assert logits.shape == (len(labels), len(classes))
        loss = nn.functional.cross_entropy(logits, labels.to(device), reduction="sum")
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite evaluation loss")
        loss_sum += loss.item()
        probabilities = logits.softmax(1).cpu()
        scores, predicted = probabilities.max(1)
        update_matrix(matrix, labels.numpy(), predicted.numpy())
        if keep_predictions:
            for offset, index in enumerate(indices):
                row = cache.rows[index]
                predictions.append({"image_id": row["image_id"], "image_path": row["output_relative_path"],
                                    "group_id": row["group_id"], "actual_index": int(labels[offset]),
                                    "actual_class": classes[int(labels[offset])], "predicted_index": int(predicted[offset]),
                                    "predicted_class": classes[int(predicted[offset])], "score": float(scores[offset]),
                                    "correct": bool(labels[offset] == predicted[offset])})
    return metrics_from_matrix(matrix, classes, loss_sum / len(cache)), predictions


def train(config, run_dir, resume=False, max_updates=None):
    run_dir = Path(run_dir)
    if resume and (run_dir / "status.json").exists():
        status = json.loads((run_dir / "status.json").read_text())
        if status.get("state") == "complete":
            print("This procedure is already complete. Preserving its checkpoints; use a new run for a new experiment.", flush=True)
            return
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = device_for(config.get("device", "auto"))
    frozen = release()
    locked = json.loads((ROOT / "data/manifests/leafcheck177_frozen.json").read_text())
    assert frozen["fingerprint"] == locked["fingerprint"], "Dataset changed after its freeze"
    classes = frozen["classes"]; size = config["input_size"]
    spec = json.loads((cache_path(size, frozen["fingerprint"]) / "preprocessing.json").read_text())
    train_data = PixelCache("train", size, frozen); validation = PixelCache("val", size, frozen)
    assert len(train_data) == 59008 and len(validation) == 16836
    seed_everything(config["seed"])
    model_config = config["model"]
    model = LeafCNN(**model_config).to(device=device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5,
                                                         patience=2, threshold=.001, threshold_mode="abs", min_lr=.00002)
    counts = np.bincount(train_data.labels, minlength=177)
    power = config.get("class_weight_power", 0.0)
    weights = np.clip((counts.mean() / counts) ** power, .2, 5.0)
    weights /= np.sum(weights * counts) / len(train_data)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                  reduction="none", label_smoothing=config.get("label_smoothing", 0.0))
    crop_loss_weight = config.get("crop_loss_weight", 0.0)
    if not math.isfinite(crop_loss_weight) or crop_loss_weight < 0:
        raise ValueError("Crop loss weight must be finite and nonnegative")
    same_crop = None
    if crop_loss_weight:
        crops = np.array([label.split("__", 1)[0] for label in classes])
        same_crop = torch.from_numpy(crops[:, None] == crops).to(device)
    history = []; best = None; epoch = 1; next_batch = 0
    matrix = np.zeros((177, 177), np.int64); loss_sum = 0.0; seen = 0
    early_score = -1.0; meaningful_epoch = 0; total_steps = 0; accumulated_seconds = 0.0; epoch_elapsed = 0.0
    if resume:
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        assert checkpoint["config"] == config and checkpoint["dataset_fingerprint"] == frozen["fingerprint"]
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]); scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = checkpoint["epoch"]; next_batch = checkpoint["next_batch"]
        matrix = np.asarray(checkpoint["train_matrix"], np.int64)
        loss_sum = checkpoint["train_loss_sum"]; seen = checkpoint["train_seen"]
        history = checkpoint["history"]; best = checkpoint["best"]
        early_score = checkpoint["early_score"]; meaningful_epoch = checkpoint["meaningful_epoch"]
        total_steps = checkpoint["total_steps"]; accumulated_seconds = checkpoint["training_seconds"]
        epoch_elapsed = checkpoint.get("epoch_elapsed_seconds", 0.0)
        restore_rng(checkpoint["rng"])
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(run_dir / "config.json", config)
        atomic_json(run_dir / "environment.json", environment(device))
        atomic_json(run_dir / "dataset.json", frozen)
        atomic_json(run_dir / "preprocessing.json", spec)
        atomic_json(run_dir / "class_to_idx.json", frozen["class_to_idx"])
        atomic_json(run_dir / "class_weights.json", dict(zip(classes, weights.tolist())))
        shutil.copytree(ROOT / "leafcheck", run_dir / "source/leafcheck", ignore=shutil.ignore_patterns("__pycache__"))
    stop = {"requested": False}
    def request_stop(signum, frame):
        stop["requested"] = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    began = time.perf_counter(); updates_this_call = 0

    def save_resume(current_epoch, offset):
        atomic_checkpoint(run_dir / "last.pt", {
            "format_version": 1, "config": config, "dataset_fingerprint": frozen["fingerprint"],
            "model_state_dict": cpu_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "rng": rng_state(), "epoch": current_epoch, "next_batch": offset,
            "train_matrix": matrix.tolist(), "train_loss_sum": loss_sum, "train_seen": seen,
            "history": history, "best": best, "early_score": early_score, "meaningful_epoch": meaningful_epoch,
            "total_steps": total_steps, "training_seconds": accumulated_seconds + time.perf_counter() - began,
            "epoch_elapsed_seconds": epoch_elapsed + time.perf_counter() - epoch_started if offset else 0.0})

    parameters = sum(p.numel() for p in model.parameters())
    print(json.dumps({"run": str(run_dir), "device": str(device), "parameters": parameters,
                      "train_images": len(train_data), "validation_images": len(validation),
                      "dataset_fingerprint": frozen["fingerprint"], "resume_epoch": epoch, "resume_batch": next_batch}), flush=True)
    batch_size = config["batch_size"]; batches = math.ceil(len(train_data) / batch_size)
    for current_epoch in range(epoch, config["max_epochs"] + 1):
        epoch_started = time.perf_counter()
        order = np.random.default_rng(config["seed"] + current_epoch * 100003).permutation(len(train_data))
        model.train(); maximum_gradient = 0.0
        for batch in range(next_batch, batches):
            indices = order[batch * batch_size:(batch + 1) * batch_size]
            raw, labels = train_data.batch(indices)
            inputs = prepare_batch(raw, spec, device, config["augmentation"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            targets = labels.to(device)
            loss = loss_fn(logits, targets).mean()
            if same_crop is not None:
                loss = loss + crop_loss_weight * crop_marginal_loss(logits, targets, same_crop).mean()
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Nonfinite training loss; last valid checkpoint is preserved")
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config.get("gradient_clip", 5.0), foreach=False, error_if_nonfinite=True)
            maximum_gradient = max(maximum_gradient, norm.item())
            optimizer.step()
            loss_sum += loss.item() * len(indices); seen += len(indices)
            update_matrix(matrix, labels.numpy(), logits.detach().argmax(1).cpu().numpy())
            total_steps += 1; updates_this_call += 1
            if (batch + 1) % 100 == 0:
                save_resume(current_epoch, batch + 1)
                print(f"epoch {current_epoch} batch {batch + 1}/{batches}; loss {loss_sum / seen:.4f}; {seen:,} images", flush=True)
            if stop["requested"] or (max_updates is not None and updates_this_call >= max_updates):
                save_resume(current_epoch, batch + 1)
                atomic_json(run_dir / "status.json", {"state": "interrupted", "epoch": current_epoch,
                            "next_batch": batch + 1, "total_steps": total_steps, "saved_utc": utc_now()})
                print("Saved resumable checkpoint at a completed update.", flush=True)
                return
        assert seen == len(train_data), "An epoch must visit every training image once"
        training = metrics_from_matrix(matrix, classes, loss_sum / seen)
        val, predictions = evaluate(model, validation, spec, device, classes, config.get("validation_batch_size", batch_size * 2), True)
        chosen = better(val, best)
        if chosen:
            best = {"macro_f1": val["macro_f1"], "loss": val["loss"], "accuracy": val["accuracy"], "epoch": current_epoch}
            payload = {"format_version": 1, "model_version": "leafcheck177-v1", "model_config": model_config,
                       "model_state_dict": cpu_state(model), "classes": classes, "class_to_idx": frozen["class_to_idx"],
                       "preprocessing": spec, "dataset_fingerprint": frozen["fingerprint"], "epoch": current_epoch,
                       "selection": "validation macro-F1; loss tie-break within 1e-6; earlier epoch", "validation": val,
                       "config": config, "parameters": parameters}
            atomic_checkpoint(run_dir / "best_model.pt", payload)
            atomic_json(run_dir / "validation_metrics.json", val)
            write_csv(run_dir / "validation_predictions.csv", predictions)
        if val["macro_f1"] > early_score + config.get("early_min_delta", .001):
            early_score = val["macro_f1"]; meaningful_epoch = current_epoch
        scheduler.step(val["macro_f1"])
        row = {"epoch": current_epoch, "train_loss": training["loss"], "train_accuracy": training["accuracy"],
               "train_macro_f1": training["macro_f1"], "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
               "validation_macro_f1": val["macro_f1"], "learning_rate_next": optimizer.param_groups[0]["lr"],
               "epoch_seconds": epoch_elapsed + time.perf_counter() - epoch_started, "train_images": seen,
               "maximum_gradient_norm": maximum_gradient, "selected": chosen}
        history.append(row); write_csv(run_dir / "history.csv", history)
        atomic_json(run_dir / f"validation_epoch_{current_epoch:03d}.json", val)
        print(json.dumps(row), flush=True)
        matrix = np.zeros((177, 177), np.int64); loss_sum = 0.0; seen = 0; next_batch = 0; epoch_elapsed = 0.0
        save_resume(current_epoch + 1, 0)
        atomic_json(run_dir / "status.json", {"state": "running", "completed_epochs": current_epoch,
                    "best": best, "total_steps": total_steps, "saved_utc": utc_now()})
        if current_epoch >= config["min_epochs"] and current_epoch - meaningful_epoch >= config["patience"]:
            reason = "early stopping after validation macro-F1 plateau"
            break
    else:
        reason = "completed declared maximum epoch budget"
    # Verify the selected artifact with a fresh model on a deterministic validation batch.
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    restored = LeafCNN(**checkpoint["model_config"]).to(device=device, memory_format=torch.channels_last)
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True); restored.eval()
    verified, _ = evaluate(restored, validation, spec, device, classes, config.get("validation_batch_size", batch_size * 2))
    assert abs(verified["macro_f1"] - checkpoint["validation"]["macro_f1"]) < 1e-12
    assert abs(verified["loss"] - checkpoint["validation"]["loss"]) < 1e-5
    final_status = {"state": "complete", "completed_epochs": current_epoch,
                "best": best, "total_steps": total_steps, "training_seconds": accumulated_seconds + time.perf_counter() - began,
                "stopping_reason": reason, "checkpoint_reload_verified": True, "saved_utc": utc_now()}
    lineage_path = run_dir / "lineage.json"
    if lineage_path.exists():
        lineage = json.loads(lineage_path.read_text())
        final_status["parent_run"] = lineage["parent_run"]
        final_status["additional_training_seconds"] = final_status["training_seconds"] - lineage["parent_training_seconds"]
    atomic_json(run_dir / "status.json", final_status)
    print("COMPLETE", json.dumps(best), reason, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-updates", type=int, help="Diagnostic interruption; produces a resumable checkpoint")
    args = parser.parse_args()
    config = json.loads((args.run / "config.json" if args.resume else args.config).read_text())
    train(config, args.run, args.resume, args.max_updates)


if __name__ == "__main__":
    main()
