"""Verify publisher originals and prepare the isolated corn training addition."""

import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np

from audit_field_preflight import protected_field_sources
from prepare_blight_dataset import Photo, inspect
from prepare_final_dataset import file_sha256, read_csv
from prepare_tom2024 import build_registered_release, check_bytes, fetch_registered_images, local_path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/manifests"
PLAN = MANIFESTS / "leafcheck_bangladesh_corn_source.json"
REGISTRY = MANIFESTS / "leafcheck_bangladesh_corn_intake_files.csv"
PRETORIA = MANIFESTS / "leafcheck_pretoria_expanded.csv"
RAW = ROOT / "data/raw/leafcheck_field_v10/bangladesh-corn"
OUTPUT = ROOT / "data/processed/leafcheck_bangladesh_corn_v1"


def protection():
    plan = json.loads((ROOT / "configs/leafcheck_pretoria_baseline_v1.json").read_text())
    if file_sha256(PRETORIA) != plan["manifest_sha256"]:
        raise ValueError("Expected the registered Pretoria development bytes")
    fingerprint, count, existing, raw, pixels, paths, phashes, offsets = protected_field_sources()
    extra_hashes, extra_offsets = [], []
    for row in read_csv(PRETORIA):
        if row["screen"] != "no_candidate":
            continue
        path = ROOT / "data/raw/leafcheck_field_v9/pretoria" / f"{int(row['source_id'])}.jpg"
        photo = Photo("pretoria", str(path), "", row["label"], row["source_label"], "val",
                      "pretoria:all", "", path.read_bytes())
        inspect(photo)
        if photo.reason or photo.sha256 != row["sha256"] or photo.pixel_hash != row["pixel_d4_sha256"]:
            raise ValueError(f"Unexpected validation bytes for {row['source_id']}")
        identifier = "pretoria:" + row["source_id"]
        raw.setdefault(photo.sha256, identifier)
        pixels.setdefault(photo.pixel_hash, identifier)
        extra_hashes.extend(photo.perceptual)
        extra_offsets.extend([len(paths)] * len(photo.perceptual))
        paths.append(identifier)
    return (fingerprint, count, existing, raw, pixels, paths,
            np.concatenate((phashes, np.array(extra_hashes, dtype=np.uint64))),
            np.concatenate((offsets, np.array(extra_offsets, dtype=np.int64))))


def build(output):
    source = json.loads(PLAN.read_text())
    counts = Counter(row["class_name"] for row in read_csv(REGISTRY))
    if counts != source["publisher_jpeg_counts"]:
        raise ValueError("The original JPEG registry changed")

    def candidate(row):
        content = local_path(row, RAW).read_bytes()
        check_bytes(content, row)
        locator = f"RAW_Image/{row['source_label']}/{row['filename']}:{row['file_id']}"
        photo = Photo("bangladesh_corn", locator, row["url"], row["class_name"], row["source_label"],
                      "train", "bangladesh_corn:collection", "", content)
        inspect(photo)
        return photo

    build_registered_release(output, REGISTRY, PLAN, RAW, "bangladesh_corn", "train", protection, candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "build"))
    parser.add_argument("--output", default=OUTPUT, type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch_registered_images(REGISTRY, RAW)
    else:
        build(args.output)
