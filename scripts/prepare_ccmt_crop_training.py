"""Restore a bounded sample of original CCMT images for crop-only training."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from build_pakistan_tomato_development import DEFAULT as PAKISTAN, protection
from prepare_final_dataset import file_sha256, read_csv, write_csv
from prepare_plantcity_holdout import DEFAULT as PLANTCITY
from prepare_tom2024 import (
    MANIFESTS, ROOT, Photo, check_bytes, deduplicate, extend_protected,
    fetch_one, filter_protected, inspect, local_path,
)


NAME = "leafcheck_ccmt_crop_v1"
INTAKE = MANIFESTS / "leafcheck_ccmt_crop_intake.json"
REGISTRY = MANIFESTS / f"{NAME}_files.csv"
RAW = ROOT / "data/raw/leafcheck_field_v6/ccmt"
OUTPUT = ROOT / "data/processed" / NAME


def inventory(source, registry):
    if registry.exists():
        raise FileExistsError(registry)

    def records(item):
        directory, (crop, publisher_condition) = item
        params = urlencode({"version": 1, "folder_id": directory})
        request = Request(f"{source['source_api']}?{params}", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            files = json.load(response)
        choices = []
        for file in files:
            details = file["content_details"]
            identifier = file["id"]
            if details["content_type"] != "image/jpeg":
                continue
            url = f"https://data.mendeley.com/public-files/datasets/bwh3zbpkpv/files/{identifier}/file_downloaded"
            choices.append({
                "file_id": identifier, "sha256": details["sha256_hash"], "size": details["size"],
                "crop": crop, "source_folder": directory, "publisher_condition": publisher_condition,
                "filename": file["filename"], "url": url,
            })
        if len(choices) < source["sample_per_folder"]:
            raise ValueError(f"Insufficient eligible catalog entries in {directory}")
        ordered = sorted(choices, key=lambda row: hashlib.sha256(row["file_id"].encode()).digest())
        return ordered[:source["sample_per_folder"]]

    with ThreadPoolExecutor(4) as executor:
        rows = [row for group in executor.map(records, sorted(source["folders"].items())) for row in group]
    if len({row["file_id"] for row in rows}) != len(rows):
        raise ValueError("Supplier catalog repeated a file ID across registered folders")
    write_csv(registry, rows)
    print(f"Pinned {len(rows)} original-folder file IDs", flush=True)


def build(source, registry, raw, output):
    if output.exists():
        raise FileExistsError(output)
    rows = read_csv(registry)
    expected = Counter({folder: source["sample_per_folder"] for folder in source["folders"]})
    if Counter(row["source_folder"] for row in rows) != expected:
        raise ValueError("Registered sample does not match the intake")

    def prepare(row):
        content = local_path(row, raw).read_bytes()
        check_bytes(content, row)
        if source["folders"][row["source_folder"]] != [row["crop"], row["publisher_condition"]]:
            raise ValueError("Registered crop or source folder changed")
        locator = f"{row['source_folder']}/{row['filename']}:{row['file_id']}"
        photo = Photo("ccmt", locator, row["url"], row["crop"], row["publisher_condition"],
                      "train", "ccmt:collection", "", content)
        inspect(photo)
        return photo

    with ThreadPoolExecutor(6) as executor:
        photos = list(executor.map(prepare, rows))
    protected = extend_protected(protection(), PAKISTAN,
                                 MANIFESTS / "leafcheck_pakistan_tomato_dev_v1_release.json")
    protected = extend_protected(protected, PLANTCITY,
                                 MANIFESTS / "leafcheck_plantcity_holdout_v1_release.json")
    filter_protected(photos, protected)
    deduplicate(photos)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".ccmt-") as stage:
        folder = Path(stage) / "release"
        folder.mkdir()
        records = []
        for photo, row in zip(photos, rows, strict=True):
            if not photo.reason:
                photo.output = f"train/{photo.class_name}/{photo.sha256}{photo.extension}"
                path = folder / photo.output
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(photo.content)
            records.append({
                "image_id": "sha256:" + photo.sha256, "source": photo.source, "crop": photo.class_name,
                "publisher_condition": row["publisher_condition"], "group_id": photo.group,
                "split": "" if photo.reason else "train",
                "status": "excluded" if photo.reason else "included",
                "reason": photo.reason, "related_to": photo.related, "source_locator": photo.locator,
                "sha256": photo.sha256, "pixel_d4_sha256": photo.pixel_hash,
                "output_relative_path": photo.output, "width": photo.width, "height": photo.height,
            })
        write_csv(folder / "manifest.csv", records)
        (folder / "source_files.csv").write_bytes(registry.read_bytes())
        hashes = {name: file_sha256(folder / name) for name in ("manifest.csv", "source_files.csv")}
        report = {
            "version": NAME, "base_release": protected[0], "candidates": len(photos),
            "role": source["role"], "intake_sha256": file_sha256(INTAKE), "files_sha256": hashes,
            "fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "crop_counts": dict(Counter(p.class_name for p in photos if not p.reason)),
            "exclusions": dict(Counter(p.reason for p in photos if p.reason)),
        }
        (folder / "release.json").write_text(json.dumps(report, indent=2) + "\n")
        folder.rename(output)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "fetch", "build"))
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    source = json.loads(INTAKE.read_text())
    with (args.raw / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "inventory":
            inventory(source, args.registry)
        elif args.action == "fetch":
            rows = read_csv(args.registry)
            with ThreadPoolExecutor(5) as executor:
                for index, _ in enumerate(executor.map(lambda row: fetch_one(row, args.raw), rows), 1):
                    if index % 50 == 0 or index == len(rows):
                        print(f"Verified {index}/{len(rows)} publisher image hashes", flush=True)
        else:
            build(source, args.registry, args.raw, args.output)


if __name__ == "__main__":
    main()
