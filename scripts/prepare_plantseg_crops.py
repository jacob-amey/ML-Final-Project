"""Reconcile PlantSeg provenance and build training-only lesion context crops."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from io import BytesIO, TextIOWrapper
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from zipfile import ZipFile
import zlib

from PIL import Image

from prepare_blight_dataset import FROZEN, ROOT
from prepare_final_dataset import file_sha256, read_csv, write_csv


NAME = "leafcheck_plantseg_regions_v1"
DEFAULT = ROOT / "data/processed" / NAME
ARCHIVE = ROOT / "runs/field_v3/plantseg_sparse.zip"
URL = "https://zenodo.org/records/17719108/files/plantseg.zip?download=1"
SIZE = 1057281724
RANGES = [
    (0, 33164502, "8384ed20062f7dac22b87e2896e5f09a5fc5b4b494b3b13177e762865887da79"),
    (35564501, 36002077, "29b746391b84222eb6a7b27a55d34b978110d7a5a91e1e9a55032bde5e5561e7"),
    (1053087420, SIZE, "3b6e4e49e92770539831e2a7017b508133344ba4e082bb7787cc74f184e0fb33"),
]


def file_ranges(archive):
    with archive.open("rb") as source:
        for start, stop, expected in RANGES:
            source.seek(start)
            if hashlib.sha256(source.read(stop - start)).hexdigest() != expected:
                raise ValueError("The local PlantSeg archive range differs from its pin")


def fetch(archive):
    if archive.exists():
        file_ranges(archive)
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=archive.parent) as work:
        scratch = Path(work)
        stage = scratch / "partial.zip"
        with stage.open("wb") as target:
            for start, stop, expected in RANGES:
                file, headers = scratch / "range", scratch / "headers"
                subprocess.run(["curl", "-fLSs", "--max-time", "240", "--max-filesize", str(stop - start),
                                "--range", f"{start}-{stop - 1}", "-D", str(headers), "-o", str(file), URL],
                               check=True)
                if f"content-range: bytes {start}-{stop - 1}/{SIZE}" not in headers.read_text().lower():
                    raise ValueError("PlantSeg response is missing the expected byte range")
                if file_sha256(file) != expected:
                    raise ValueError("PlantSeg publisher byte range changed")
                target.seek(start)
                target.write(file.read_bytes())
        stage.rename(archive)


def members(archive):
    with archive.open("plantseg/Metadata.csv") as content:
        metadata = list(csv.DictReader(TextIOWrapper(content, encoding="utf-8-sig")))
    lookup = {Path(item.filename).name.lower(): item for item in archive.infolist()
              if item.filename.endswith((".jpg", ".png"))}
    originals = {Path(row["source_relative_path"]).name.lower(): row for row in read_csv(FROZEN / "manifest.csv")
                 if Path(row["source_relative_path"]).name.lower() in lookup}
    if len(metadata) != len(originals) or len(metadata) != 7774:
        raise ValueError("PlantSeg provenance does not match the recorded master's membership")
    return [(detail, originals[detail["Name"].lower()], lookup[detail["Name"].lower()],
             lookup[detail["Label file"].lower()]) for detail in metadata]


def match_original(entry, master):
    _, row, image, _ = entry
    source = (master / row["master_relative_path"]).read_bytes()
    if (len(source) != image.file_size or zlib.crc32(source) != image.CRC
            or hashlib.sha256(source).hexdigest() != row["sha256"]):
        raise ValueError(f"Archive and frozen master disagree on {row['source_relative_path']}")
    return source


def audit(archive, entries, master):
    with ThreadPoolExecutor(5) as pool:
        verified = sum(pool.map(lambda entry: bool(match_original(entry, master)), entries))
    return {
        "publisher_records": len(entries),
        "matched_crc_size_and_frozen_sha256": verified,
        "leafcheck_roles": dict(Counter(row["split"] or "excluded" for _, row, _, _ in entries)),
        "publisher_test_leafcheck_roles": dict(Counter(row["split"] or "excluded" for detail, row, _, _ in entries
                                                       if detail["Split"] == "Test")),
        "publisher_metadata_sha256": hashlib.sha256(archive.read("plantseg/Metadata.csv")).hexdigest(),
        "archive": URL,
        "method": "Case-insensitive image basename, archive ZIP size and CRC32, original file SHA-256.",
        "release_use": "Already present in the original frozen master. Never an independent test or new image source.",
    }


def normalize(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def box_with_context(bounds, width, height):
    def interval(low, high, size):
        span = min(size, max(1.8 * (high - low), .55 * size))
        start = max(0, min(size - span, .5 * (low + high - span)))
        return math.floor(start), math.ceil(start + span)

    left, top, right, bottom = bounds
    l, r = interval(left, right, width)
    t, b = interval(top, bottom, height)
    return l, t, r, b


def crop_input(archive, entry, master):
    detail, row, _, mask_member = entry
    source, mark = match_original(entry, master), archive.read(mask_member)
    with Image.open(BytesIO(source)) as image, Image.open(BytesIO(mark)) as mask:
        if image.getexif().get(274, 1) != 1 or image.size != mask.size or mask.mode != "L":
            return None
        values = {i for i, count in enumerate(mask.histogram()) if count}
        if values != {0, int(detail["Index"]) + 1}:
            return None
        crop = box_with_context(mask.getbbox(), *image.size)
        if crop == (0, 0, *image.size):
            return None
        buffer = BytesIO()
        image.convert("RGB").crop(crop).save(buffer, "PNG")
    return crop, buffer.getvalue(), hashlib.sha256(mark).hexdigest()


def build(archive, entries, master, output, base_fingerprint):
    if output.exists():
        raise FileExistsError(output)
    candidates = defaultdict(list)
    for entry in entries:
        detail, row, _, _ = entry
        plant = normalize(detail["Plant"])
        condition = normalize(detail["Disease"]).removeprefix(plant + "_")
        if (detail["Split"] == "Training" and row["split"] == "train" and row["status"] == "included"
                and detail["License"] in {"CC-BY-NC", "CC0"} and f"{plant}__{condition}" == row["class_name"]):
            candidates[row["class_name"]].append(entry)
    records, used_contents = [], set()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".plantseg-") as directory:
        stage = Path(directory) / "release"
        stage.mkdir()
        for class_name, choices in sorted(candidates.items()):
            accepted = 0
            for entry in sorted(choices, key=lambda x: hashlib.sha256(f"{NAME}:{x[1]['image_id']}".encode()).hexdigest()):
                result = crop_input(archive, entry, master)
                if result is None:
                    continue
                crop, content, mask_sha256 = result
                digest = hashlib.sha256(content).hexdigest()
                if digest in used_contents:
                    continue
                used_contents.add(digest)
                detail, row, image_member, mask_member = entry
                path = f"train/{class_name}/{digest}.png"
                target = stage / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                records.append(dict(image_id=f"sha256:{digest}", sha256=digest, source="plantseg",
                                    class_name=class_name, class_id=row["class_id"], split="train", status="included",
                                    group_id=row["group_id"], source_locator=image_member.filename,
                                    source_url=detail["URL"], source_license=detail["License"],
                                    parent_image_id=row["image_id"], mask_locator=mask_member.filename,
                                    mask_sha256=mask_sha256, crop=",".join(map(str, crop)), output_relative_path=path))
                accepted += 1
                if accepted == 12:
                    break
        write_csv(stage / "manifest.csv", records)
        source = {
            "name": NAME, "doi": "10.5281/zenodo.17719108", "archive": URL, "ranges": RANGES,
            "citation": "Tianqi Wei et al., A Large-Scale In-the-wild Dataset for Plant Disease Segmentation.",
            "selection": "Exact canonical label, both designated training splits, at most 12 deterministic parent IDs per label.",
            "geometry": "Mask bounding box enlarged to 180% per dimension with a minimum of 55% of the original dimension.",
            "role": "Derived training views of frozen master parents. No addition to independent original counts.",
        }
        (stage / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        seal = {"version": NAME, "base_release": base_fingerprint,
                "files_sha256": {name: file_sha256(stage / name) for name in ("manifest.csv", "source.json")},
                "train_derived_counts": dict(Counter(row["class_name"] for row in records)),
                "sources": dict(Counter(row["source_license"] for row in records))}
        seal["fingerprint"] = hashlib.sha256(json.dumps(seal["files_sha256"], sort_keys=True).encode()).hexdigest()
        (stage / "release.json").write_text(json.dumps(seal, indent=2) + "\n")
        stage.rename(output)
    print(json.dumps({"derived_training_views": len(records), "labels": len(set(x["class_name"] for x in records)),
                      "fingerprint": seal["fingerprint"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "audit", "build"))
    parser.add_argument("--archive", default=ARCHIVE, type=Path)
    parser.add_argument("--output", default=DEFAULT, type=Path)
    parser.add_argument("--master", type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.archive)
    else:
        file_ranges(args.archive)
        frozen = json.loads((ROOT / "data/manifests/leafcheck177_frozen.json").read_text())
        if any(file_sha256(FROZEN / name) != expected for name, expected in frozen["files"].items()):
            raise ValueError("Frozen master metadata differs from the existing release")
        master = args.master or Path(json.loads((FROZEN / "dataset_info.json").read_text())["source_master"])
        with ZipFile(args.archive) as source:
            entries = members(source)
            if args.action == "audit":
                print(json.dumps(audit(source, entries, master), indent=2))
            else:
                build(source, entries, master, args.output, frozen["fingerprint"])
