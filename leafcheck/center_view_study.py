"""Screen a predetermined central view on validation without fitting model weights."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import torch

from .common import ROOT, atomic_json, sha256, utc_now
from .crop_head_study import measures
from .data import PixelCache, release as master_release
from .field_expansion import load_inputs
from .inference import Predictor
from .potato_policy import distribution, master_probabilities
from .train import device_for
from .upload_study import ARTIFACT, pixels, read


PLAN = ROOT / "configs/leafcheck_center_view_v1.json"
DATA = ROOT / "data/processed"


def centered_pixels(path, size, fraction):
    with Image.open(path) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    width, height = original.size
    dx = round((1 - fraction) * width / 2)
    dy = round((1 - fraction) * height / 2)
    crop = original.crop((dx, dy, width - dx, height - dy))
    view = ImageOps.pad(crop, (size, size), method=Image.Resampling.BILINEAR, color=(128, 128, 128))
    return np.frombuffer(view.tobytes(), np.uint8).reshape(size, size, 3)


def center_probability(predictor, records, plan):
    size = predictor.spec["input_size"]
    def decode(record):
        path = record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Validation image changed: {path}")
        return centered_pixels(path, size, plan["center_fraction_each_axis"])
    inferred = []
    with ThreadPoolExecutor(6) as pool:
        for offset in range(0, len(records), 128):
            items = records[offset:offset + 128]
            raw = torch.from_numpy(np.stack(list(pool.map(decode, items))))
            inferred.append(distribution(predictor.scoring_model, raw, predictor.spec, predictor.device))
    return np.concatenate(inferred)


def run(output, device):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    clock = time.monotonic()
    plan = read(PLAN)
    freeze = master_release()
    predictor = Predictor(ARTIFACT, device)
    if freeze["fingerprint"] != plan["master_fingerprint"] or predictor.metadata["checkpoint_sha256"] != plan["reference_checkpoint"]:
        raise ValueError("Base identities differ from the registered protocol")
    lineage = {"initial_weights": plan["reference_checkpoint"]}
    _, v2, first, second = load_inputs(DATA / "leafcheck_field_v2_sampled", lineage)
    _, pk, _, pakistan = load_inputs(DATA / "leafcheck_pakistan_tomato_dev_v1", lineage)
    field = {
        name: [r for r in first + second if r["source"] == name and r["split"] == "val"]
        for name in plan["field_sources"] if name != "pakistan"
    }
    field["pakistan"] = pakistan
    if not all(field.values()) or any(r["split"] != "val" for group in field.values() for r in group):
        raise ValueError("Only the three registered development groups may be scored")
    cache = PixelCache("val", predictor.spec["input_size"], freeze)
    master = [dict(r, path=DATA / "leafcheck177_final" / r["output_relative_path"]) for r in cache.rows]
    mix = plan["probability_weight"]
    comparisons = {}
    for name, samples in {**field, "master": master}.items():
        if name == "master":
            originals = master_probabilities(predictor, cache)
        else:
            originals = distribution(predictor.scoring_model, pixels(samples, predictor.spec["input_size"]),
                                     predictor.spec, predictor.device)
        center = center_probability(predictor, samples, plan)
        labels = np.array([int(r["class_id"]) for r in samples])
        comparisons[name] = {
            "reference": measures(originals, labels, predictor.classes),
            "center_blend": measures((1 - mix) * originals + mix * center, labels, predictor.classes),
        }
        print(name, "reference", comparisons[name]["reference"]["macro_f1"],
              "center blend", comparisons[name]["center_blend"]["macro_f1"], flush=True)
    differences = {
        name: {key: m["center_blend"][key] - m["reference"][key] for key in ("macro_f1", "accuracy", "wrong_crop")}
        for name, m in comparisons.items()
    }
    gains = [differences[name]["macro_f1"] for name in field]
    mean_gain = float(np.mean(gains))
    passes = (
        mean_gain >= plan["min_field_joint_macro_f1_gain"]
        and min(gains) >= -plan["max_field_source_drop"]
        and differences["master"]["macro_f1"] >= -plan["max_master_joint_macro_f1_drop"]
        and differences["master"]["accuracy"] >= -plan["max_master_accuracy_drop"]
    )
    atomic_json(output, {
        "protocol_sha256": sha256(PLAN), "checkpoint_sha256": plan["reference_checkpoint"],
        "master_fingerprint": freeze["fingerprint"], "field_fingerprints": [v2["fingerprint"], pk["fingerprint"]],
        "completed_utc": utc_now(), "device": str(device), "seconds": time.monotonic() - clock,
        "comparisons": comparisons, "differences": differences, "field_gain": mean_gain,
        "passes_validation_screen": passes, "device_latency_check_pending_if_passed": passes, "test_opened": False,
    })
    print("Center view screen:", mean_gain, passes, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, device_for("auto"))
