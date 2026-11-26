"""Fetch a bounded sample from the publisher's corn archive for provenance review."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from zipfile import ZipFile

from audit_field_preflight import protected_field_sources
from inspect_pakistan_tomato import fetch_range
from prepare_blight_dataset import Photo, deduplicate, filter_protected, inspect, protected_hashes
from prepare_final_dataset import file_sha256, write_csv


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/manifests/leafcheck_pretoria_corn_intake.json"
RAW = ROOT / "data/raw/leafcheck_field_v9/pretoria"
COLUMNS = ("GLS", "NCLB", "PLS", "CR", "SR", "NoFoliarSymptoms", "Other", "UnidentifiedDisease")


def prepare(output, expanded=False):
    source = json.loads(PLAN.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    size = source["archive_size"]
    start = size - source["directory_bytes"]
    tail = RAW / "zip-tail.bin"
    metadata = RAW / "Database.csv"
    if not metadata.exists():
        subprocess.run(["curl", "-fsSL", "--max-time", "25", source["metadata_url"], "-o", str(metadata)], check=True)
    if not tail.exists():
        fetch_range(source, start, size, tail)
    if file_sha256(metadata) != source["metadata_sha256"] or file_sha256(tail) != source["directory_sha256"]:
        raise ValueError("Publisher metadata or archive directory changed")

    with tempfile.TemporaryDirectory(dir=RAW) as work:
        index = Path(work) / "archive.zip"
        with index.open("wb") as stream:
            stream.seek(start)
            stream.write(tail.read_bytes())
        with ZipFile(index) as archive:
            members = sorted(archive.infolist(), key=lambda m: m.header_offset)
        stops = {a.filename: b.header_offset for a, b in zip(members, members[1:])}
        stops[members[-1].filename] = start
        by_name = defaultdict(list)
        for member in members:
            name = PurePosixPath(member.filename)
            if not member.is_dir() and name.suffix.casefold() in {".jpg", ".jpeg", ".png"}:
                by_name[name.name.casefold()].append(member)
        with metadata.open(encoding="utf-8-sig", newline="") as stream:
            labels = list(csv.DictReader(stream))
        if len(labels) != 2355 or len({row["imgID_id"] for row in labels}) != len(labels):
            raise ValueError("The publisher's image ID table changed")
        candidates = defaultdict(list)
        mixed = 0
        for row in labels:
            matching = by_name[row["filePath"].casefold()]
            positive = [label for label in COLUMNS if row[label] == "1"]
            mixed += len(positive) > 1
            if len(matching) == len(positive) == 1 and positive[0] in source["mapping"]:
                candidates[positive[0]].append((row["imgID_id"], matching[0], positive[0]))
        per_class = 24 if expanded else 4
        queue = [value for group in candidates.values() for value in sorted(group, key=lambda r:
                 sha256(("leafcheck-pretoria-intake-v1:" + r[0]).encode()).hexdigest())[:per_class]]
        ranges = RAW / "ranges"
        ranges.mkdir(exist_ok=True)

        def fetch(entry):
            identifier, member, _ = entry
            offset = member.header_offset
            path = ranges / f"{int(identifier)}.bin"
            if not path.exists():
                stage = path.with_suffix(".part")
                fetch_range(source, offset, stops[member.filename], stage)
                stage.replace(path)
            return offset, path

        with ThreadPoolExecutor(4) as pool, index.open("r+b") as stream:
            for offset, path in pool.map(fetch, queue):
                stream.seek(offset)
                stream.write(path.read_bytes())
        photos = []
        with ZipFile(index) as archive:
            for identifier, member, label in queue:
                raw = archive.read(member)  # ZipFile checks the publisher's CRC.
                photo = Photo("pretoria", member.filename, source["archive_url"], source["mapping"][label],
                              label, "candidate", "pretoria:all", "", raw)
                inspect(photo)
                (RAW / f"{int(identifier)}.jpg").write_bytes(raw)
                photo.content = b""
                photos.append(photo)
    filter_protected(photos, protected_field_sources() if expanded else protected_hashes())
    deduplicate(photos)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "sample.csv"
    write_csv(prediction_path, [{
        "source_id": identifier, "member": photo.locator, "label": photo.class_name,
        "source_label": label, "offset": member.header_offset, "compressed_size": member.compress_size,
        "sha256": photo.sha256, "pixel_d4_sha256": photo.pixel_hash, "camera_date": photo.date,
        "camera": photo.device.rstrip("\0").strip(), "screen": photo.reason or "no_candidate",
    } for (identifier, member, label), photo in zip(queue, photos, strict=True)])
    result = {
        "source_plan_sha256": file_sha256(PLAN), "audited_sample_sha256": file_sha256(prediction_path),
        "published_members": len(labels), "multiple_condition_records": mixed,
        "compatible_single_label_candidate_counts": {source["mapping"][key]: len(rows)
                                                    for key, rows in candidates.items()},
        "sample_photos": len(queue), "sample_screen_counts": Counter(p.reason or "no_candidate" for p in photos),
        "exif_capture_days": Counter(p.date.split()[0] for p in photos if p.date),
        "split_assigned": False, "photos_admitted": 0, "inference_run": False,
    }
    if expanded:
        result["overlap_scope"] = "Frozen master and registered prior field releases"
        result["capture_days_by_compatible_label"] = {
            label: Counter(p.date.split()[0] for p in photos if p.class_name == label and p.date and not p.reason)
            for label in source["mapping"].values()
        }
        result["unknown_capture_date"] = sum(not p.date for p in photos)
    (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expanded", action="store_true")
    args = parser.parse_args()
    prepare(args.output, args.expanded)
