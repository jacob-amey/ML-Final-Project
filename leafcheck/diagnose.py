"""Training-only forward, gradient, tiny-learning, reload and throughput checks."""

import argparse
import copy
import json
import time
import numpy as np
import torch
from torch import nn
from .common import ROOT, RUNS, atomic_checkpoint, atomic_json, environment, seed_everything
from .data import PixelCache, cache_path, release
from .model import LeafCNN
from .preprocessing import prepare_batch
from .train import cpu_state, device_for


def synchronize(device):
    if device.type == "mps": torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    torch.set_num_threads(4); seed_everything(434)
    frozen = release(); data = PixelCache("train", args.size, frozen)
    spec = json.loads((cache_path(args.size, frozen["fingerprint"]) / "preprocessing.json").read_text())
    device = device_for(args.device); out = RUNS / "diagnostics"; out.mkdir(exist_ok=True, parents=True)
    indices = []
    for label in [0, 30, 60, 90, 120, 150]:
        indices.extend(np.flatnonzero(data.labels == label)[:2])
    raw, labels = data.batch(indices); x = prepare_batch(raw, spec, device); y = labels.to(device)
    model = LeafCNN(architecture="residual_v1", dropout=0).to(device=device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
    first = next(model.parameters()).detach().clone(); loss_fn = nn.CrossEntropyLoss()
    history = []
    for step in range(1, 121):
        model.train(); opt.zero_grad(set_to_none=True); logits = model(x)
        assert logits.shape == (12, 177)
        loss = loss_fn(logits, y); assert torch.isfinite(loss).item(); loss.backward()
        if step == 1:
            assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
        opt.step()
        if step % 20 == 0:
            model.eval()
            with torch.no_grad():
                result = model(x); correct = int((result.argmax(1) == y).sum().item()); measured = loss_fn(result, y).item()
            history.append({"updates": step, "loss": measured, "correct": correct, "total": 12})
            print("Tiny learning:", history[-1], flush=True)
            if correct == 12 and measured < .05: break
    assert not torch.equal(first, next(model.parameters()).detach())
    assert history[-1]["correct"] == 12 and history[-1]["loss"] < .1, "Tiny pipeline failed to learn"
    model.eval()
    with torch.no_grad(): reference = model(x).cpu()
    atomic_checkpoint(out / "tiny.pt", {"state": cpu_state(model)})
    reloaded = LeafCNN(architecture="residual_v1", dropout=0).to(device=device, memory_format=torch.channels_last)
    reloaded.load_state_dict(torch.load(out / "tiny.pt", map_location="cpu", weights_only=True)["state"]); reloaded.eval()
    with torch.no_grad(): torch.testing.assert_close(reloaded(x).cpu(), reference, rtol=1e-5, atol=1e-5)
    results = []
    del model, reloaded, opt
    for architecture in ("plain_v1", "residual_v1"):
        for batch in (128, 256):
            raw, targets = data.batch(np.arange(batch)); inputs = prepare_batch(raw, spec, device, "dihedral")
            model = LeafCNN(architecture=architecture).to(device=device, memory_format=torch.channels_last)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
            targets = targets.to(device)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True); loss_fn(model(inputs), targets).backward(); optimizer.step()
            synchronize(device); start = time.perf_counter()
            for step in range(12):
                optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(inputs), targets); loss.backward(); optimizer.step()
            synchronize(device); seconds = time.perf_counter() - start
            result = {"architecture": architecture, "batch_size": batch, "input_size": args.size,
                      "seconds_per_update": seconds / 12, "images_per_second": batch * 12 / seconds,
                      "parameters": sum(p.numel() for p in model.parameters()), "finite_loss": bool(torch.isfinite(loss).item())}
            results.append(result); print("Benchmark:", result, flush=True)
            del model, optimizer, inputs
            if device.type == "mps": torch.mps.empty_cache()
    atomic_json(out / "checks.json", {"environment": environment(device), "dataset_fingerprint": frozen["fingerprint"],
                "tiny_history": history, "gradient_and_update_verified": True, "checkpoint_reload_verified": True,
                "benchmark": results, "benchmark_conditions": "Warm MPS, float32, channels_last, cached training images; 3 warmup + 12 timed optimizer updates; excludes validation and checkpoint I/O."})


if __name__ == "__main__":
    main()
