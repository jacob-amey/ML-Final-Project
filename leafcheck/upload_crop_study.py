"""Evaluate a fixed light-background framing change on development data only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .common import ROOT, atomic_json, sha256
from .inference import Predictor
from .preprocessing import image_bytes
from .train import device_for
from .upload_study import ARTIFACT, SAVED_VALIDATION, load_study, probabilities, read, rows, score, summarize


def framed_pixels(path, side):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    preview = image.copy()
    preview.thumbnail((256, 256))
    probe = np.asarray(preview).astype(np.int16)
    bg = (probe.min(2) >= 170) & (probe.max(2) - probe.min(2) <= 45)
    borders = (bg[:4], bg[-4:], bg[:, :4], bg[:, -4:])
    if min(float(border.mean()) for border in borders) < .85:
        return None
    yy, xx = np.where(~bg)
    if not .04 <= len(xx) / bg.size <= .70:
        return None
    x0, x1 = np.quantile(xx, [.005, .995])
    y0, y1 = np.quantile(yy, [.005, .995])
    height, width = bg.shape
    dx, dy = (x1 - x0) * .18, (y1 - y0) * .18
    box = (max(0, (x0 - dx) / width), max(0, (y0 - dy) / height),
           min(1, (x1 + dx + 1) / width), min(1, (y1 + dy + 1) / height))
    if (box[2] - box[0]) * (box[3] - box[1]) > .85:
        return None
    crop = image.crop(tuple(int(c * (image.width if i % 2 == 0 else image.height)) for i, c in enumerate(box)))
    return np.asarray(ImageOps.pad(crop, (side, side), method=Image.Resampling.BILINEAR, color=(128, 128, 128)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    args = parser.parse_args()
    protocol = ROOT / "configs/upload_paper_crop_v1.json"
    frozen, master, supplement = load_study(read(ROOT / "configs/upload_blight_v1.json"))
    val = [r for r in master if r["split"] == "val"]
    extra = [r for r in supplement if r["split"] == "val"]
    evaluation = val + extra
    predict = {r["image_id"]: r["predicted_index"] for r in rows(SAVED_VALIDATION)}
    classes = frozen["classes"]
    recorded = {r["image_id"]: classes.index(r["predicted"]) for r in rows(args.input / "incumbent_scored_rows.csv")
                if r["source"] in {"sheikh", "banu", "bugwood"}}
    true = np.array([int(r["class_id"]) for r in evaluation])
    base = np.array([int(predict[r["image_id"]]) for r in val] + [recorded[r["image_id"]] for r in extra])
    device = device_for(args.device)
    torch.set_num_threads(4)
    evaluator = Predictor(ARTIFACT, device)
    active, crops, originals = [], [], []
    size = evaluator.spec["input_size"]

    def prepare(r):
        return framed_pixels(r["path"], size)

    with ThreadPoolExecutor(6) as pool:
        for i, pixels in enumerate(pool.map(prepare, evaluation)):
            if pixels is not None:
                record = evaluation[i]
                assert sha256(record["path"]) == record["sha256"]
                active.append(i)
                crops.append(pixels)
                raw = image_bytes(record["path"], size)
                originals.append(np.frombuffer(raw, np.uint8).reshape(size, size, 3))
    assert active
    crop_p = probabilities(evaluator.scoring_model, torch.from_numpy(np.stack(crops)), evaluator.spec, device)
    old_p = probabilities(evaluator.scoring_model, torch.from_numpy(np.stack(originals)), evaluator.spec, device)
    assert np.array_equal(old_p.argmax(1), base[active])
    splits = {"master": np.arange(len(val)),
              "supplement": np.array([len(val) + i for i, r in enumerate(extra) if r["source_slice"] != "bugwood"]),
              "field": np.array([len(val) + i for i, r in enumerate(extra) if r["source_slice"] == "bugwood"])}
    result = {"recipe_hash": sha256(protocol), "eligible_images": len(active), "metrics": {}}
    for name, scores in (("incumbent", old_p), ("crop_only", crop_p), ("mean_original_crop", (old_p + crop_p) / 2)):
        changed = base.copy()
        changed[active] = scores.argmax(1)
        result["metrics"][name] = {}
        for split, idx in splits.items():
            result["metrics"][name][split] = score(true[idx], changed[idx], classes, None if split == "master" else np.unique(true[idx]))
            print(name, split, summarize(result["metrics"][name][split]), flush=True)
    atomic_json(args.input / "paper_crop.json", result)


if __name__ == "__main__":
    main()
