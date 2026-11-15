"""Audit and prepare the master plant dataset with reproducible, grouped splits.

Run with the project's Python environment. The master images are read-only inputs.
The final dataset is published after all copied files and split assignments verify.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import time
import urllib.request
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageStat


PROJECT = Path(__file__).resolve().parents[1]
MASTER = PROJECT / "Plant_Dataset_Updated_2026-11-11"
WORK = PROJECT / "data/processed/.leafcheck177_preparation"
OUTPUT = PROJECT / "data/processed/leafcheck177_final"
HF_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
PLANTDOC_REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"
SEED = 434
MIN_SIDE = 32

# The first eight DCT frequencies are sufficient for a 64-bit perceptual hash.
positions = np.arange(32, dtype=np.float64)
DCT_BASIS = np.cos(np.pi / 32 * (positions + 0.5) * np.arange(8)[:, None])
DCT_BASIS[0] *= 1 / math.sqrt(2)


def read_csv(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_metadata():
    directory = WORK / "metadata"
    directory.mkdir(parents=True, exist_ok=True)
    sources = {
        "leaf-map.json": (
            "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/"
            f"{HF_REVISION}/leaf_grouping/leaf-map.json"
        ),
        "plantdoc-tree.json": (
            "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/git/trees/"
            f"{PLANTDOC_REVISION}?recursive=1"
        ),
    }
    records = []
    for name, url in sources.items():
        path = directory / name
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "LeafCheck-dataset-preparation"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        json.loads(path.read_text())
        records.append({"file": name, "url": url, "sha256": file_sha256(path)})
    (directory / "sources.json").write_text(json.dumps(records, indent=2))


def dihedral_views(array):
    """Return the eight exact rotations/reflections as NumPy views."""
    transposed = array.swapaxes(0, 1)
    return [
        array,
        array[:, ::-1],
        array[::-1, :],
        array[::-1, ::-1],
        transposed,
        transposed[:, ::-1],
        transposed[::-1, :],
        transposed[::-1, ::-1],
    ]


def symmetric_positions(length):
    first = {0, length // 8, length // 4, length // 3, length // 2}
    return sorted(first | {length - 1 - value for value in first})


def canonical_pixel_hash(image):
    """Select a canonical orientation using exact, symmetric pixel samples.

    Tied sample signatures are resolved with full pixel hashes. This preserves
    exact rotation/flip invariance without hashing eight full-size images normally.
    """
    array = np.asarray(image)
    height, width = array.shape[:2]
    grid = array[np.ix_(symmetric_positions(height), symmetric_positions(width))]
    views = dihedral_views(array)
    grid_views = dihedral_views(grid)
    signatures = [
        (view.shape[0], view.shape[1], grid_view.tobytes())
        for view, grid_view in zip(views, grid_views)
    ]
    smallest = min(signatures)
    candidates = []
    for view, signature in zip(views, signatures):
        if signature == smallest:
            digest = hashlib.sha256(f"{view.shape[0]}x{view.shape[1]}|".encode())
            digest.update(view.tobytes())
            candidates.append(digest.hexdigest())
    return min(candidates)


def perceptual_hashes(image):
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coefficients = DCT_BASIS @ gray @ DCT_BASIS.T
    signs = (-1.0) ** np.arange(8)
    transformed = [
        coefficients,
        coefficients * signs[None, :],
        coefficients * signs[:, None],
        coefficients * signs[:, None] * signs[None, :],
        coefficients.T,
        coefficients.T * signs[None, :],
        coefficients.T * signs[:, None],
        coefficients.T * signs[:, None] * signs[None, :],
    ]
    values = []
    for transformed_coefficients in transformed:
        flat = transformed_coefficients.ravel()
        bits = np.packbits(flat > np.median(flat[1:]))
        values.append(int.from_bytes(bits.tobytes(), "big"))
    return list(dict.fromkeys(values))


def audit_image(row):
    record = {"master_path": row["output_relative_path"], "errors": []}
    try:
        path = (MASTER / row["output_relative_path"]).resolve()
        if not path.is_relative_to(MASTER.resolve()):
            raise ValueError("Master path escapes the dataset root")
        data = path.read_bytes()
        record["actual_sha256"] = hashlib.sha256(data).hexdigest()
        record["git_blob_sha1"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if record["actual_sha256"] != row["sha256"]:
            record["errors"].append("checksum_mismatch")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as original:
                original.seek(0)
                original.load()
                record.update({
                    "width": original.width,
                    "height": original.height,
                    "format": original.format,
                    "mode": original.mode,
                    "frames": getattr(original, "n_frames", 1),
                    "exif_orientation": original.getexif().get(274, 1),
                })
                image = ImageOps.exif_transpose(original).convert("RGB")
                record["pixel_d4_sha256"] = canonical_pixel_hash(image)
                record["phashes"] = perceptual_hashes(image)
                extrema = image.getextrema()
                record["uniform"] = all(low == high for low, high in extrema)
                record["gray_std"] = ImageStat.Stat(image.convert("L")).stddev[0]
                if min(image.size) < MIN_SIDE:
                    record["errors"].append("minimum_side_below_32")
                if record["uniform"]:
                    record["errors"].append("uniform_image")
        if not row["source_relative_path"] or not row["source_class_name"] or not row["group_id"]:
            record["errors"].append("missing_source_or_group_record")
        if Path(row["output_relative_path"]).parts[:2] != (row["split"], row["class_name"]):
            record["errors"].append("manifest_label_path_mismatch")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        record["errors"].append("decode_or_file_error")
        record["error_detail"] = f"{type(error).__name__}: {error}"
    return record


def audit(workers):
    WORK.mkdir(parents=True, exist_ok=True)
    fetch_metadata()
    rows = read_csv(MASTER / "manifest.csv")
    snapshot = {
        "manifest_sha256": file_sha256(MASTER / "manifest.csv"),
        "class_mapping_sha256": file_sha256(MASTER / "class_to_idx.json"),
        "minimum_side": MIN_SIDE,
        "audit_version": 1,
    }
    snapshot_path = WORK / "snapshot.json"
    if snapshot_path.exists():
        assert json.loads(snapshot_path.read_text()) == snapshot, "Audit cache belongs to another input or policy"
    else:
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
    cache_path = WORK / "image_audit.jsonl"
    cached = {}
    if cache_path.exists():
        with cache_path.open() as file:
            for line in file:
                record = json.loads(line)
                cached[record["master_path"]] = record
    pending = [row for row in rows if row["output_relative_path"] not in cached]
    print(f"Images already audited {len(cached):,}; remaining {len(pending):,}", flush=True)
    started = time.monotonic()
    with cache_path.open("a") as file, ThreadPoolExecutor(max_workers=workers) as pool:
        for index, record in enumerate(pool.map(audit_image, pending), start=1):
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
            if index % 1000 == 0 or index == len(pending):
                file.flush()
                print(f"Decoded {len(cached) + index:,}/{len(rows):,} images; {time.monotonic() - started:.0f}s this pass", flush=True)
    print("Pixel decoding and checksums completed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["audit"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    audit(args.workers)


if __name__ == "__main__":
    main()
