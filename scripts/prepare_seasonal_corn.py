"""Restore a bounded training-only sample from the pinned seasonal corn archive."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import ZipFile

from inspect_pakistan_tomato import fetch_range
from prepare_bangladesh_corn import protection
from prepare_blight_dataset import Photo, inspect
from prepare_final_dataset import file_sha256, write_csv
from prepare_tom2024 import build_registered_release, check_bytes, local_path


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data/manifests"
PIN = MANIFESTS / "leafcheck_seasonal_corn_intake.json"
SOURCE = MANIFESTS / "leafcheck_seasonal_corn_training_source.json"
RAW = ROOT / "data/raw/leafcheck_field_v10/seasonal-corn"
OUTPUT = ROOT / "data/processed/leafcheck_seasonal_corn_v1"


def acquire():
    pin = json.loads(PIN.read_text())
    source = json.loads(SOURCE.read_text())
    boundary = pin["archive_size"] - pin["directory_bytes"]
    tail = RAW / "index.bin"
    if not tail.exists():
        fetch_range(pin, boundary, pin["archive_size"], tail)
    if file_sha256(tail) != pin["directory_sha256"]:
        raise ValueError("Unexpected archive directory bytes")
    ranges = RAW / "ranges"
    restore = RAW / "originals"
    ranges.mkdir(exist_ok=True)
    restore.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory(dir=RAW) as work:
        archive_path = Path(work) / "archive.zip"
        with archive_path.open("wb") as file:
            file.seek(boundary)
            file.write(tail.read_bytes())
        with ZipFile(archive_path) as archive:
            members = sorted(archive.infolist(), key=lambda m: m.header_offset)
        stops = {a.filename: b.header_offset for a, b in zip(members, members[1:])}
        stops[members[-1].filename] = pin["archive_size"]
        groups = defaultdict(list)
        for member in members:
            path = PurePosixPath(member.filename)
            if (len(path.parts) == 3 and path.parts[0] == "Final corn dataset"
                    and path.parts[1] in source["class_map"] and path.suffix.lower() in {".jpg", ".jpeg", ".png"}):
                groups[path.parts[1]].append(member)
        if {key: len(images) for key, images in groups.items()} != pin["candidate_counts"]:
            raise ValueError("The source folder counts changed")
        chosen = [
            (key, member) for key, images in sorted(groups.items())
            for member in sorted(images, key=lambda m: sha256(("seasonal-intake-v1:" + m.filename).encode()).hexdigest())[
                :source["source_counts"][key]]
        ]

        def retrieve(member):
            start = member.header_offset
            stop = min(stops[member.filename], boundary)
            if start >= stop:
                return None
            path = ranges / f"{start}.bin"
            if not path.exists():
                temporary = path.with_suffix(".part")
                fetch_range(pin, start, stop, temporary)
                temporary.replace(path)
            return start, path

        with ThreadPoolExecutor(5) as pool, archive_path.open("r+b") as file:
            for number, part in enumerate(pool.map(retrieve, [member for _, member in chosen]), 1):
                if part is not None:
                    start, path = part
                    file.seek(start)
                    file.write(path.read_bytes())
                if number % 50 == 0 or number == len(chosen):
                    print("Archive members available", number, "of", len(chosen), flush=True)
        records = []
        with ZipFile(archive_path) as archive:
            for key, member in chosen:
                content = archive.read(member)
                identifier = str(member.header_offset)
                digest = sha256(content).hexdigest()
                (restore / f"{identifier}.bin").write_bytes(content)
                records.append({
                    "file_id": identifier, "source_folder": key, "filename": member.filename,
                    "class_name": source["class_map"][key], "sha256": digest, "size": len(content),
                    "url": pin["archive_url"],
                })
    registry = RAW / "training-originals.csv"
    write_csv(registry, records)
    print("Verified source archive image records", len(records), flush=True)
    return registry, restore


def build(output):
    registry, restore = acquire()

    def candidate(row):
        content = local_path(row, restore).read_bytes()
        check_bytes(content, row)
        photo = Photo("seasonal_corn", row["filename"], row["url"], row["class_name"],
                      row["source_folder"], "train", "seasonal_corn:collection", "", content)
        inspect(photo)
        return photo

    build_registered_release(output, registry, SOURCE, restore, "seasonal_corn", "train", protection, candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT, type=Path)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    with (RAW / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        build(args.output)
