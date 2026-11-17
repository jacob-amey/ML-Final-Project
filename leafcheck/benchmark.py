"""Benchmark configured training updates on train pixels, between accelerator jobs.

This fixed-batch software check measures resources, never disease accuracy.
"""

import argparse
import gc
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .common import (ROOT, atomic_json, environment, hardware, seed_everything,
                     update_matrix, utc_now)
from .data import PixelCache, cache_path, release
from .model import LeafCNN
from .preprocessing import prepare_batch
from .train import device_for


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--updates", type=int, default=12)
    args = parser.parse_args()
    assert args.warmup >= 1 and args.updates >= 1
    frozen = release()
    assert frozen["fingerprint"] == json.loads((ROOT / "data/manifests/leafcheck177_frozen.json").read_text())["fingerprint"]
    records = []
    for path in args.configs:
        config = json.loads(path.read_text())
        torch.set_num_threads(config.get("cpu_threads", 4)); seed_everything(config["seed"])
        device = device_for(config["device"])
        data = PixelCache("train", config["input_size"], frozen)
        spec = json.loads((cache_path(config["input_size"], frozen["fingerprint"]) / "preprocessing.json").read_text())
        assert spec["statistics_split"] == "train"
        indices = np.random.default_rng(config["seed"]).choice(len(data), config["batch_size"], replace=False)
        raw, targets = data.batch(indices)
        model = LeafCNN(**config["model"]).to(device=device, memory_format=torch.channels_last)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        counts = np.bincount(data.labels, minlength=177)
        weights = np.clip((counts.mean() / counts) ** config.get("class_weight_power", 0), .2, 5)
        weights /= np.sum(weights * counts) / len(data)
        loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                            reduction="none", label_smoothing=config.get("label_smoothing", 0))
        matrix = np.zeros((177, 177), np.int64)
        first = next(model.parameters()).detach().clone()
        model.train(); started = None
        for step in range(args.warmup + args.updates):
            if step == args.warmup:
                synchronize(device); started = time.perf_counter()
            inputs = prepare_batch(raw, spec, device, config["augmentation"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            assert logits.shape == (len(targets), 177)
            loss = loss_fn(logits, targets.to(device)).mean()
            assert torch.isfinite(loss).item()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], foreach=False, error_if_nonfinite=True)
            norm.item(); optimizer.step()
            update_matrix(matrix, targets.numpy(), logits.detach().argmax(1).cpu().numpy())
        synchronize(device); seconds = time.perf_counter() - started
        assert not torch.equal(first, next(model.parameters()).detach())
        memory = {"allocated_tensor_bytes":torch.mps.current_allocated_memory(),
                  "driver_allocated_bytes_including_cache":torch.mps.driver_allocated_memory(),
                  "recommended_max_working_set_bytes":torch.mps.recommended_max_memory()} if device.type == "mps" else {}
        record = {"config_file":str(path), "config":config, "device":str(device),
                  "parameters":sum(p.numel() for p in model.parameters()),
                  "seconds_per_update":seconds / args.updates,
                  "images_per_second":len(targets) * args.updates / seconds,
                  "warmup_updates":args.warmup, "timed_updates":args.updates,
                  "finite_loss_and_gradient_check":True, "parameter_update_verified":True,
                  "mps_memory_after_timed_updates":memory, "hardware":hardware(),
                  "free_disk_bytes":shutil.disk_usage(ROOT).free,
                  "environment":environment(device)}
        records.append(record)
        atomic_json(args.output, {"created_utc":utc_now(), "dataset_fingerprint":frozen["fingerprint"],
                                 "split":"train", "benchmarks":records,
                                 "conditions":"Sequential float32 channels-last updates on one fixed real training batch. Includes augmentation/normalization, weighted loss, backward, clipping, AdamW and metric transfers; excludes decode/cache setup, checkpoint I/O and full validation. Memory is a post-update allocator snapshot, not a measured peak. No accuracy evidence or test predictions."})
        print(json.dumps({k:record[k] for k in ["config_file", "parameters", "seconds_per_update", "images_per_second", "mps_memory_after_timed_updates"]}), flush=True)
        del model, optimizer, first, inputs, logits, loss, loss_fn, data, raw, targets
        gc.collect()
        if device.type == "mps": torch.mps.empty_cache()


if __name__ == "__main__": main()
