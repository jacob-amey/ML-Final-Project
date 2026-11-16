"""Test full-training-set BatchNorm statistics after weights have stopped changing."""

import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch import nn
from .common import ROOT, atomic_checkpoint, atomic_json, environment, sha256, write_csv
from .data import PixelCache, release
from .model import LeafCNN
from .preprocessing import prepare_batch
from .train import cpu_state, device_for, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(); parent = args.parent; directory = args.run
    parent_status = json.loads((parent / "status.json").read_text())
    assert parent_status["state"] == "complete" and parent_status["checkpoint_reload_verified"]
    assert not directory.exists(), "A refinement run cannot be overwritten"
    checkpoint = torch.load(parent / "best_model.pt", map_location="cpu", weights_only=True)
    frozen = release(); assert checkpoint["dataset_fingerprint"] == frozen["fingerprint"]
    config = {**checkpoint["config"], "experiment": parent.name + "-bn-training-statistics",
              "reference": str(parent), "refinement": "reestimate_batchnorm_statistics",
              "hypothesis": "Full-training-set deterministic feature statistics may reduce noise in the last-batch running statistics. Only normalization buffers change; learned weights stay fixed."}
    directory.mkdir(parents=True)
    atomic_json(directory / "config.json", config)
    atomic_json(directory / "environment.json", environment("CPU statistics; MPS validation when available"))
    for name in ["dataset.json", "class_to_idx.json", "class_weights.json", "preprocessing.json", "history.csv"]:
        shutil.copyfile(parent / name, directory / name)
    shutil.copytree(ROOT / "leafcheck", directory / "source/leafcheck", ignore=shutil.ignore_patterns("__pycache__"))
    torch.set_num_threads(4)
    spec = checkpoint["preprocessing"]; classes = checkpoint["classes"]
    train = PixelCache("train", spec["input_size"], frozen); val = PixelCache("val", spec["input_size"], frozen)
    model = LeafCNN(**checkpoint["model_config"]).to(memory_format=torch.channels_last)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    original_parameters = {name:p.detach().clone() for name,p in model.named_parameters()}
    device = device_for(config["device"])
    model.to(device)
    reference, _ = evaluate(model, val, spec, device, classes, config["validation_batch_size"])
    assert abs(reference["macro_f1"] - checkpoint["validation"]["macro_f1"]) < 1e-12
    model.to("cpu"); model.eval()
    batch_norms = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    old_momentum = [module.momentum for module in batch_norms]
    for module in batch_norms:
        module.reset_running_stats(); module.momentum = None; module.train()
    order = np.random.default_rng(config["seed"]).permutation(len(train))
    seen = 0; started = time.perf_counter(); batch_size = config["batch_size"]
    with torch.no_grad():
        for start in range(0, len(train), batch_size):
            indices = order[start:start + batch_size]; raw, _ = train.batch(indices)
            model(prepare_batch(raw, spec, "cpu"))
            seen += len(indices)
            if (start // batch_size + 1) % 100 == 0 or seen == len(train):
                print(f"Training-only BatchNorm statistics: {seen:,}/{len(train):,} images; {time.perf_counter() - started:.1f}s", flush=True)
    assert seen == 59008
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.detach(), original_parameters[name], atol=0, rtol=0)
    for module, momentum in zip(batch_norms, old_momentum):
        module.momentum = momentum; module.eval()
    model.to(device)
    metrics, predictions = evaluate(model, val, spec, device, classes, config["validation_batch_size"], True)
    payload = {**checkpoint, "model_state_dict":cpu_state(model), "config":config, "validation":metrics,
               "refinement": {"method":"reestimate_batchnorm_statistics", "source":"train", "images":seen,
                              "parent_checkpoint_sha256":sha256(parent / "best_model.pt"), "learned_parameters_changed":False}}
    atomic_checkpoint(directory / "best_model.pt", payload)
    restored = LeafCNN(**payload["model_config"]).to(device=device, memory_format=torch.channels_last)
    restored.load_state_dict(torch.load(directory / "best_model.pt", weights_only=True, map_location="cpu")["model_state_dict"])
    verified, _ = evaluate(restored, val, spec, device, classes, config["validation_batch_size"])
    assert abs(verified["macro_f1"]-metrics["macro_f1"]) < 1e-12
    atomic_json(directory / "validation_metrics.json", metrics); write_csv(directory / "validation_predictions.csv", predictions)
    seconds = time.perf_counter()-started
    atomic_json(directory / "refinement.json", {"parent":str(parent), "parent_checkpoint_sha256":sha256(parent / "best_model.pt"),
                "reference_validation_macro_f1":reference["macro_f1"], "refined_validation_macro_f1":metrics["macro_f1"],
                "difference":metrics["macro_f1"]-reference["macro_f1"], "additional_seconds":seconds,
                "learned_parameters_unchanged":True, "normalization_images":seen, "test_predictions_computed":False})
    atomic_json(directory / "status.json", {"state":"complete", "completed_epochs":parent_status["completed_epochs"],
                "best":{"macro_f1":metrics["macro_f1"], "accuracy":metrics["accuracy"], "loss":metrics["loss"], "epoch":checkpoint["epoch"]},
                "training_seconds":parent_status["training_seconds"]+seconds, "additional_refinement_seconds":seconds,
                "stopping_reason":"Completed training-only BatchNorm statistics pass after the parent's completed training procedure",
                "parent_run":str(parent), "checkpoint_reload_verified":True})
    print(json.dumps({"reference_macro_f1":reference["macro_f1"], "refined_macro_f1":metrics["macro_f1"], "difference":metrics["macro_f1"]-reference["macro_f1"]},indent=2))


if __name__ == "__main__": main()
