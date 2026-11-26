"""Probe whether sparse terminal-map responses change field classification."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common import ROOT, atomic_json, sha256, utc_now
from .data import release
from .evaluate_feature_style import development_pixels, development_rows
from .field_expansion import field_scores
from .inference import Predictor
from .scoring import PredictionScores
from .train import device_for
from .upload_study import ARTIFACT, read


PLAN = ROOT / "configs/leafcheck_spatial_pool_probe_v1.json"


class RootMeanSquareProbe(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        maps = self.model.features(inputs)
        average = maps.mean(dim=(2, 3))
        pooled = maps.square().mean(dim=(2, 3)).sqrt()
        scale = average.sum(1, keepdim=True) / pooled.sum(1, keepdim=True).clamp_min(1e-12)
        return self.model.classifier[-1](pooled * scale)


def compare(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    plan, frozen = read(PLAN), release()
    checkpoint = plan["checkpoint_sha256"]
    if sha256(ARTIFACT / "model.pt") != checkpoint or frozen["fingerprint"] != plan["dataset"]:
        raise ValueError("Probe reference changed")
    device = device_for("auto")
    base = Predictor(ARTIFACT, device)
    members, fingerprints = development_rows(checkpoint)
    dev = development_pixels(members, base.spec["input_size"], frozen["class_to_idx"])
    models = {"average": base.model, "spatial_probe": RootMeanSquareProbe(base.model)}
    both = {}
    for name, model in models.items():
        scorer = PredictionScores(model, {"name": "d4_probability_mean"}).to(device).eval()
        both[name] = field_scores(scorer, dev, base.classes, base.spec, device, 32)
        print(name, {k: v["macro_f1"] for k, v in both[name].items()}, flush=True)
    expected = read(ROOT / plan["reference_scores"])["measurements"]["deployed"]["fields"]
    for source in expected:
        actual = both["average"][source]
        if any(expected[source][key] != actual[key] for key in ("correct", "wrong_crop", "sample_count", "macro_f1")):
            raise ValueError(f"The original field scores changed for {source}")
    changes = {source: both["spatial_probe"][source]["macro_f1"] - before["macro_f1"]
               for source, before in both["average"].items()}
    mean_change = float(np.mean(list(changes.values())))
    passed = mean_change >= plan["min_field_macro_f1_gain"] and min(changes.values()) >= -plan["max_source_macro_f1_drop"]
    atomic_json(output, {"evaluated_utc": utc_now(), "plan_sha256": sha256(PLAN), "checkpoint_sha256": checkpoint,
                         "development_fingerprints": fingerprints, "results": both, "source_gains": changes,
                         "mean_field_macro_f1_gain": mean_change, "passed_probe": passed,
                         "test_opened": False, "deployment_changed": False, "device": str(device)})
    print("Mean field change", mean_change, "passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    compare(parser.parse_args().output)
