"""Frozen release metadata and deterministic, fingerprinted RGB pixel caches."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from .common import DATA, ROOT, atomic_json, sha256, write_csv
from .preprocessing import PREPROCESSING_VERSION, image_bytes


def release():
    mapping = json.loads((DATA / "class_to_idx.json").read_text())
    classes = sorted(mapping, key=mapping.get)
    assert len(classes) == 177 and list(mapping.values()) == list(range(177))
    names = ["manifest.csv", "class_to_idx.json", "split_summary.csv", "preparation_snapshot.json"]
    files = {name: sha256(DATA / name) for name in names}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"version": "leafcheck177_final-v1", "fingerprint": fingerprint, "files": files,
            "classes": classes, "class_to_idx": mapping, "root": str(DATA)}


def split_rows(split):
    with (DATA / "manifest.csv").open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if row["status"] == "included" and row["split"] == split]
    return sorted(rows, key=lambda row: row["output_relative_path"])


def cache_path(size, fingerprint):
    return ROOT / f"data/processed/leafcheck177_cache/{fingerprint[:16]}/{PREPROCESSING_VERSION}_{size}"


def build_cache(split, size, frozen, workers=6):
    directory = cache_path(size, frozen["fingerprint"]); directory.mkdir(parents=True, exist_ok=True)
    metadata_path = directory / f"{split}.json"
    target = directory / f"{split}.npy"
    if metadata_path.exists() and target.exists():
        meta = json.loads(metadata_path.read_text())
        assert meta["dataset_fingerprint"] == frozen["fingerprint"] and meta["input_size"] == size
        assert sha256(target) == meta["pixels_sha256"], "Pixel cache changed; rebuild this cache."
        return
    rows = split_rows(split)
    temporary = target.with_suffix(".tmp.npy")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=(len(rows), size, size, 3))
    started = time.perf_counter()

    def prepare(row):
        path = DATA / row["output_relative_path"]
        assert sha256(path) == row["sha256"], "An image changed after the release audit"
        return image_bytes(path, size)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, pixels in enumerate(pool.map(prepare, rows)):
            array[index] = np.frombuffer(pixels, np.uint8).reshape(size, size, 3)
            if (index + 1) % 5000 == 0 or index + 1 == len(rows):
                print(f"{split} cache: {index + 1:,}/{len(rows):,}; {time.perf_counter() - started:.1f}s", flush=True)
    array.flush(); del array; temporary.replace(target)
    atomic_json(metadata_path, {"dataset_fingerprint": frozen["fingerprint"], "input_size": size,
                               "preprocessing_version": PREPROCESSING_VERSION, "pixels_sha256": sha256(target),
                               "count": len(rows), "rows": [{key: row[key] for key in (
                                   "image_id", "class_id", "class_name", "output_relative_path", "group_id", "sha256")}
                                   for row in rows]})


class PixelCache:
    def __init__(self, split, size, frozen, allow_test=False):
        if split not in {"train", "val"} and not (split == "test" and allow_test):
            raise ValueError("Development code cannot open the test cache.")
        directory = cache_path(size, frozen["fingerprint"])
        self.metadata = json.loads((directory / f"{split}.json").read_text())
        assert self.metadata["dataset_fingerprint"] == frozen["fingerprint"]
        assert self.metadata["preprocessing_version"] == PREPROCESSING_VERSION
        assert sha256(directory / f"{split}.npy") == self.metadata["pixels_sha256"], "Pixel cache integrity check failed"
        self.pixels = np.load(directory / f"{split}.npy", mmap_mode="r")
        assert self.pixels.dtype == np.uint8 and self.pixels.shape == (self.metadata["count"], size, size, 3)
        self.rows = self.metadata["rows"]
        self.labels = np.asarray([int(row["class_id"]) for row in self.rows], dtype=np.int64)
        assert len(self.pixels) == len(self.labels)
        assert all(frozen["classes"][int(row["class_id"])] == row["class_name"] for row in self.rows)

    def __len__(self):
        return len(self.labels)

    def batch(self, indices):
        return torch.from_numpy(np.array(self.pixels[indices], copy=True)), torch.from_numpy(self.labels[indices].copy())


def training_statistics(cache):
    total = 0; sums = np.zeros(3, np.float64); squared = np.zeros(3, np.float64)
    for start in range(0, len(cache), 256):
        x = np.asarray(cache.pixels[start:start + 256], dtype=np.float64) / 255
        sums += x.sum(axis=(0, 1, 2)); squared += np.square(x).sum(axis=(0, 1, 2))
        total += int(np.prod(x.shape[:3]))
    mean = sums / total; std = np.sqrt(squared / total - np.square(mean))
    assert (std > 0).all()
    return {"version": PREPROCESSING_VERSION, "input_size": cache.metadata["input_size"],
            "mean": mean.tolist(), "std": std.tolist(), "statistics_split": "train",
            "statistics_images": len(cache), "resize": "Pillow ImageOps.pad bilinear, RGB padding 128",
            "orientation": "EXIF transpose", "channels": "RGB", "frame": 0,
            "scaling": "uint8 / 255, then (x - mean) / std", "tensor_layout": "NCHW"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    frozen = release()
    out = ROOT / "data/manifests/leafcheck177_frozen.json"
    if out.exists():
        assert json.loads(out.read_text())["fingerprint"] == frozen["fingerprint"], "Frozen dataset changed"
    atomic_json(out, frozen)
    for split in ("train", "val"):
        build_cache(split, args.size, frozen, args.workers)
    train = PixelCache("train", args.size, frozen)
    spec = training_statistics(train)
    atomic_json(cache_path(args.size, frozen["fingerprint"]) / "preprocessing.json", spec)
    counts = np.bincount(train.labels, minlength=177)
    majority = int(counts.argmax())
    atomic_json(ROOT / "data/manifests/leafcheck177_rule.json", {
        "rule": "Always predict the most frequent training class", "class_index": majority,
        "class_name": frozen["classes"][majority], "training_count": int(counts[majority]),
        "training_images": len(train), "dataset_fingerprint": frozen["fingerprint"]})
    print("Frozen fingerprint:", frozen["fingerprint"], flush=True)
    print("Training-only preprocessing:", json.dumps(spec), flush=True)


if __name__ == "__main__":
    main()
