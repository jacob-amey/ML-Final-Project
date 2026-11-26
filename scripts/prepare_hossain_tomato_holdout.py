"""Restore and screen the three-class Hossain tomato confirmation source."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

from prepare_crop_confirmation import RELEASE as CROP_CONFIRMATION
from prepare_imagesafari_crop import OUTPUT_DIR as IMAGESAFARI
from prepare_plantcity_holdout import DEFAULT as PLANTCITY, PAKISTAN
from prepare_ccmt_crop_training import OUTPUT as CCMT
from build_pakistan_tomato_development import protection
from prepare_final_dataset import read_csv, write_csv
from prepare_tom2024 import (
    MANIFESTS, ROOT, Photo, build_registered_release, check_bytes, extend_protected, fetch_one, inspect, local_path,
)


NAME = "leafcheck_hossain_tomato_v1"
INTAKE = MANIFESTS / "leafcheck_hossain_tomato_intake.json"
REGISTRY = MANIFESTS / f"{NAME}_files.csv"
RAW = ROOT / "data/raw/leafcheck_field_v8/hossain"
OUTPUT = ROOT / "data/processed" / NAME


def inventory(source, registry):
    if registry.exists():
        raise FileExistsError(registry)
    entries = []
    for folder, (source_label, label, expected) in sorted(source["folders"].items()):
        params = urlencode({"version": 1, "folder_id": folder})
        request = Request(f"{source['source_api']}?{params}", headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            files = json.load(response)
        rows = [row for row in files if row["content_details"]["content_type"] == "image/jpeg"]
        if len(rows) != expected:
            raise ValueError(f"Unexpected publisher count for {source_label}")
        for row in sorted(rows, key=lambda file: (file["filename"], file["id"])):
            identifier = row["id"]
            data = row["content_details"]
            url = f"https://data.mendeley.com/public-files/datasets/93h9p62kg4/files/{identifier}/file_downloaded"
            entries.append({"file_id": identifier, "folder": folder, "filename": row["filename"],
                            "class_name": label, "sha256": data["sha256_hash"], "size": data["size"], "url": url})
    if len({row["file_id"] for row in entries}) != len(entries):
        raise ValueError("A source ID appears under multiple folders")
    write_csv(registry, entries)
    print("Registered publisher image files:", len(entries), flush=True)


def fetch(registry, raw):
    records = read_csv(registry)
    with ThreadPoolExecutor(6) as pool:
        for index, _ in enumerate(pool.map(lambda row: fetch_one(row, raw), records), 1):
            if index % 50 == 0 or index == len(records):
                print(f"Verified {index}/{len(records)} publisher images", flush=True)


def build(source, registry, raw, output):
    folders = source["folders"]
    records = read_csv(registry)
    expected = {folder: line[2] for folder, line in folders.items()}
    if Counter(row["folder"] for row in records) != expected:
        raise ValueError("The registry does not match the preregistered folder counts")

    def photograph(row):
        source_label, label, _ = folders[row["folder"]]
        content = local_path(row, raw).read_bytes()
        check_bytes(content, row)
        if row["class_name"] != label:
            raise ValueError("The registry label differs from its source folder")
        locator = f"{source_label}/{row['filename']}:{row['file_id']}"
        photo = Photo("hossain", locator, row["url"], label, source_label, "test", "hossain:all_capture_sites", "", content)
        inspect(photo)
        return photo

    def prior():
        base = protection()
        for release in (PAKISTAN, PLANTCITY, CCMT, CROP_CONFIRMATION, IMAGESAFARI):
            base = extend_protected(base, release, MANIFESTS / f"{release.name}_release.json")
        return base

    build_registered_release(output, registry, INTAKE, raw, "hossain", "test", prior, prepare_candidate=photograph)


if __name__ == "__main__":
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
            fetch(args.registry, args.raw)
        else:
            build(source, args.registry, args.raw, args.output)
