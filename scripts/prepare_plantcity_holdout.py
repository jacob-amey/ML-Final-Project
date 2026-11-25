"""Acquire and screen the registered PlantCity confirmation source without scoring it."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

from build_pakistan_tomato_development import DEFAULT as PAKISTAN
from build_pakistan_tomato_development import protection as previously_protected
from inspect_pakistan_tomato import fetch_range
from prepare_tom2024 import MANIFESTS, ROOT, Photo, build_registered_release, extend_protected, inspect, write_csv
from prepare_final_dataset import file_sha256, read_csv


SOURCE = MANIFESTS / "leafcheck_plantcity_intake.json"
RAW = ROOT / "data/raw/leafcheck_field_v5/plantcity"
DEFAULT = ROOT / "data/processed/leafcheck_plantcity_holdout_v1"


def acquire(raw, source, preflight):
    start = source["archive_size"] - source["directory_bytes"]
    tail = raw / "index.bin"
    if not tail.exists():
        stage = raw / "index.download"
        fetch_range(source, start, source["archive_size"], stage)
        stage.replace(tail)
    if file_sha256(tail) != source["directory_sha256"]:
        raise ValueError("Publisher archive index changed")
    with tempfile.TemporaryDirectory(dir=raw) as directory:
        sparse = Path(directory) / "index.zip"
        with sparse.open("wb") as target:
            target.seek(start)
            target.write(tail.read_bytes())
        with ZipFile(sparse) as archive:
            members = sorted(archive.infolist(), key=lambda member: member.header_offset)
        stops = {current.filename: following.header_offset for current, following in zip(members, members[1:])}
        by_label = defaultdict(list)
        for member in members:
            path = Path(member.filename)
            if path.parts[0] == "test" and len(path.parts) == 3 and path.suffix.casefold() in (".jpg", ".jpeg", ".png"):
                by_label[path.parent.name].append(member)
        if sum(map(len, by_label.values())) != 10667:
            raise ValueError("The directory no longer matches the published original count")
        quota = 2 if preflight else source["sample_per_class"]
        picks = []
        for folder, (_, expected) in sorted(source["labels"].items()):
            candidates = by_label[folder]
            if len(candidates) != expected:
                raise ValueError(f"Original count mismatch for {folder}")
            picks.extend(sorted(candidates, key=lambda member: hashlib.sha256(member.filename.encode()).hexdigest())[:quota])

        def packed(member):
            offset = member.header_offset
            path = raw / "ranges" / str(offset)
            if not path.exists():
                stage = path.with_suffix(".download")
                fetch_range(source, offset, stops[member.filename], stage)
                stage.replace(path)
            return offset, path

        (raw / "ranges").mkdir(exist_ok=True)
        with ThreadPoolExecutor(4) as pool:
            with sparse.open("r+b") as target:
                for count, (offset, path) in enumerate(pool.map(packed, picks), 1):
                    target.seek(offset)
                    target.write(path.read_bytes())
                    if count % 40 == 0:
                        print(f"Acquired {count}/{len(picks)} pinned views", flush=True)
        records = []
        (raw / "images").mkdir(exist_ok=True)
        with ZipFile(sparse) as archive:
            for member in picks:
                data = archive.read(member)
                digest = hashlib.sha256(data).hexdigest()
                (raw / "images" / f"{digest}.bin").write_bytes(data)
                folder = Path(member.filename).parent.name
                records.append({
                    "file_id": digest, "member": member.filename, "source_folder": folder,
                    "class_name": source["labels"][folder][0], "sha256": digest, "size": len(data),
                })
        registry = raw / ("preflight.csv" if preflight else "selected.csv")
        write_csv(registry.with_suffix(".tmp"), records)
        registry.with_suffix(".tmp").replace(registry)
    print(f"Verified {len(records)} member checksums", flush=True)


def build(raw, output, source):
    registry = raw / "selected.csv"
    if len(read_csv(registry)) != source["sample_per_class"] * len(source["labels"]):
        raise ValueError("Expected the full registered sample")

    def candidate(row):
        content = (raw / "images" / (row["file_id"] + ".bin")).read_bytes()
        folder = Path(row["member"]).parent.name
        label = source["labels"][folder][0]
        if hashlib.sha256(content).hexdigest() != row["sha256"] or row["class_name"] != label:
            raise ValueError(f"Member changed: {row['member']}")
        photo = Photo("plantcity", row["member"], source["source"], label, folder, "test",
                      "plantcity:confirmation_source", "", content)
        inspect(photo)
        return photo

    def protected():
        return extend_protected(previously_protected(), PAKISTAN,
                                MANIFESTS / "leafcheck_pakistan_tomato_dev_v1_release.json")

    build_registered_release(output, registry, SOURCE, raw, "plantcity", "test", protected, prepare_candidate=candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "fetch", "build"))
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    with (args.raw / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "build":
            build(args.raw, args.output, source)
        else:
            acquire(args.raw, source, args.action == "preflight")
