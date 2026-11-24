"""Build train-only field additions and a separate publisher-held tomato test."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from prepare_blight_dataset import (
    FROZEN, MANIFESTS, ROOT, Photo, deduplicate, filter_protected, inspect, protected_hashes,
)
from prepare_field_holdout import include_blight
from prepare_final_dataset import file_sha256, perceptual_hashes, read_csv, write_csv


NAME = "leafcheck_field_v2"
PIN = MANIFESTS / f"{NAME}_sources.json"
RAW = ROOT / "data/raw" / NAME
PROTECTED_FIELD = ROOT / "data/processed/leafcheck_field_v1"
IMAGES = {".jpg", ".jpeg", ".png"}


def digest(path, method):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, method).hexdigest()


def fetch(tomato_only):
    sources = json.loads(PIN.read_text())["sources"]
    selected = ("bapari", "huang") if tomato_only else ("bapari", "huang", "mduma")
    RAW.mkdir(parents=True, exist_ok=True)
    for name in selected:
        for entry in sources[name]["files"]:
            path = RAW / entry["name"]
            temporary = path.with_suffix(path.suffix + ".part")
            method = "sha256" if "sha256" in entry else "md5"
            if not path.exists():
                subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                                "--continue-at", "-", "--max-time", "3600",
                                "--output", str(temporary), entry["url"]], check=True)
                if digest(temporary, method) != entry[method]:
                    raise ValueError(f"Downloaded archive differs from publisher checksum: {path.name}")
                temporary.rename(path)
            elif digest(path, method) != entry[method]:
                raise ValueError(f"Local archive differs from publisher checksum: {path.name}")
            print("Verified", path.name, flush=True)


def archive_rows(source, entry, base, pattern, labels, role, scratch):
    # BSD tar can read the unchanged publisher RAR and 7z archives on macOS.
    local = RAW / entry["name"]
    subprocess.run(["bsdtar", "-xf", str(local), "-C", str(scratch), *pattern], check=True)
    for path in sorted((scratch / base).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGES or path.name.startswith("."):
            continue
        relative = path.relative_to(scratch / base)
        # Huang provides resized originals separately from their synthetic siblings.
        folder = str(relative.parent) if source == "bapari" else relative.parts[1]
        if folder in labels:
            yield Photo(source, path.relative_to(scratch).as_posix(), entry["url"], labels[folder], folder,
                        role, source + ":collection", "", path.read_bytes())


def potato_rows(source, entry):
    local = RAW / entry["name"]
    with ZipFile(local) as archive:
        members = [r for r in archive.infolist()
                   if re.fullmatch(r"(earlyblt|lateblt|healthy)/[a-z]+\d+\.jpe?g",
                                   r.filename, re.IGNORECASE)]
        if not members:
            raise ValueError(f"No original image names found in {local}")
        selected = sorted(members, key=lambda r: hashlib.sha256(
            f"{NAME}:{r.filename}".encode()).digest())[:source["sample_per_class"]]
        for record in sorted(selected, key=lambda r: r.header_offset):
            yield Photo("mduma", f"{entry['name']}:{record.filename}", entry["url"], entry["label"],
                        entry["label"].split("__")[1], "train", "mduma:collection", "", archive.read(record))


def protected_with_field():
    frozen, count, ext, raw, exact, locators, visuals, offsets = include_blight(protected_hashes())
    metadata = json.loads((MANIFESTS / "leafcheck_field_v1_release.json").read_text())
    assert file_sha256(PROTECTED_FIELD / "manifest.csv") == metadata["manifest_sha256"]
    extra_hashes, extra_offsets = [], []
    for row in read_csv(PROTECTED_FIELD / "manifest.csv"):
        if row["status"] != "included":
            continue
        locator = "pldd-up:" + row["image_id"]
        path = PROTECTED_FIELD / row["path"]
        assert file_sha256(path) == row["sha256"]
        raw.setdefault(row["sha256"], locator)
        exact.setdefault(row["decoded_d4_sha256"], locator)
        with Image.open(path) as image:
            hashes = perceptual_hashes(ImageOps.exif_transpose(image).convert("RGB"))
        extra_offsets.extend([len(locators)] * len(hashes))
        extra_hashes.extend(hashes)
        locators.append(locator)
    return (frozen, count, ext, raw, exact, locators, np.concatenate((visuals, np.array(extra_hashes, np.uint64))),
            np.concatenate((offsets, np.array(extra_offsets, np.int64))))


def inspect_source(photo):
    inspect(photo)
    if not photo.reason and (photo.width * photo.height > 24_000_000 or len(photo.content) > 10 * 1024 * 1024):
        photo.reason = "above_app_upload_limit"
    if photo.source == "mduma":
        date = re.match(r"20[12]\d[:-]\d\d[:-]\d\d", photo.date)
        if date:
            day = date[0].replace(":", "-")
            photo.group = f"mduma:day:{day}"
            photo.object_id = f"mduma:{photo.date}:{photo.device}"
    return photo


def assign_potato_validation(candidates):
    """Use entire photographed days, with support for each of the three labels."""
    groups = defaultdict(Counter)
    for photo in candidates:
        if photo.source == "mduma" and photo.group.startswith("mduma:day:") and not photo.reason:
            groups[photo.group][photo.class_name] += 1
    labels = {p.class_name for p in candidates if p.source == "mduma"}
    total = sum(groups.values(), Counter())
    # Greedy hash order, take about 20% per label and keep a train majority.
    target = {label: min(45, int(total[label] * .20)) for label in labels}
    counts, selected = Counter(), set()
    for group in sorted(groups, key=lambda g: hashlib.sha256(f"{NAME}:{g}".encode()).digest()):
        updated = counts + groups[group]
        if (any(counts[label] < target[label] and groups[group][label] for label in labels)
                and all(updated[label] <= total[label] * .30 for label in labels)):
            selected.add(group)
            counts = updated
    if labels and min(counts[label] for label in labels) >= 12:
        for photo in candidates:
            if photo.group in selected:
                photo.split = "val"


def build(output, tomato_only):
    if output.exists():
        raise FileExistsError(f"Preserving existing release at {output}")
    protocol = json.loads(PIN.read_text())
    sources = protocol["sources"]
    selected = ("bapari", "huang") if tomato_only else ("bapari", "huang", "mduma")
    archive_hashes = {}
    for name in selected:
        for file in sources[name]["files"]:
            path = RAW / file["name"]
            method = "sha256" if "sha256" in file else "md5"
            if digest(path, method) != file[method]:
                raise ValueError(f"Publisher checksum mismatch for {path.name}")
            archive_hashes[path.name] = file_sha256(path) if method != "sha256" else file["sha256"]
    mapping = json.loads((FROZEN / "class_to_idx.json").read_text())
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".field-expansion-") as temp:
        temporary = Path(temp)
        bapari, huang = (sources[name] for name in ("bapari", "huang"))
        candidates = list(archive_rows("bapari", bapari["files"][0], "TOMATO_LEAF_DATASET", [],
                                       bapari["labels"], "train", temporary))
        candidates.extend(archive_rows("huang", huang["files"][0], "taiwan/Preprocessed data",
                                       ["taiwan/Preprocessed data"], huang["labels"], "test", temporary))
        if "mduma" in selected:
            for entry in sources["mduma"]["files"]:
                candidates.extend(potato_rows(sources["mduma"], entry))
        with ThreadPoolExecutor(6) as pool:
            candidates = list(pool.map(inspect_source, candidates))
        print(f"Decoded {len(candidates):,} candidates", flush=True)
        protected = protected_with_field()
        assert protected[0] == protocol["base_release"]
        filter_protected(candidates, protected)
        deduplicate(candidates)
        assign_potato_validation(candidates)
        stage = temporary / "release"
        stage.mkdir()
        rows = []
        for p in candidates:
            if not p.reason:
                p.output = f"{p.split}/{p.class_name}/{p.sha256}{p.extension}"
                basename = stage / p.output
                basename.parent.mkdir(parents=True, exist_ok=True)
                basename.write_bytes(p.content)
            source = sources[p.source]
            rows.append(dict(
                image_id=f"sha256:{p.sha256}", class_name=p.class_name, class_id=mapping[p.class_name],
                split=p.split if not p.reason else "", planned_split=p.split,
                group_id=p.group, object_id=p.object_id, capture_date=p.date, device=p.device,
                source=p.source, source_label=p.source_label, source_locator=p.locator,
                source_page=source["page"], source_file_url=p.url, credit=source["credit"],
                license=source["license"], license_url=source["license_url"],
                status="excluded" if p.reason else "included", reason=p.reason, related_to=p.related,
                sha256=p.sha256, pixel_d4_sha256=p.pixel_hash, width=p.width, height=p.height,
                output_relative_path=p.output))
        write_csv(stage / "manifest.csv", rows)
        shutil.copyfile(PIN, stage / "sources.json")
        shutil.copyfile(FROZEN / "class_to_idx.json", stage / "class_to_idx.json")
        hashes = {name: file_sha256(stage / name) for name in ("manifest.csv", "sources.json", "class_to_idx.json")}
        stamp = {
            "version": NAME + ("_tomato" if tomato_only else ""),
            "fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "files_sha256": hashes,
            "base_release": protected[0],
            "source_archive_sha256": archive_hashes,
            "candidate_count": len(candidates),
            "counts": [dict(source=source, split=split, label=label, count=n)
                       for (source, split, label), n in sorted(Counter(
                           (p.source, p.split, p.class_name) for p in candidates if not p.reason).items())],
            "exclusions": dict(Counter(p.reason for p in candidates if p.reason)),
        }
        (stage / "release.json").write_text(json.dumps(stamp, indent=2) + "\n")
        verify(stage)
        stage.rename(output)
    print(json.dumps(stamp, indent=2), flush=True)


def verify(path):
    stamp = json.loads((path / "release.json").read_text())
    assert all(file_sha256(path / name) == digest for name, digest in stamp["files_sha256"].items())
    assert stamp["fingerprint"] == hashlib.sha256(json.dumps(stamp["files_sha256"], sort_keys=True).encode()).hexdigest()
    mapping = json.loads((path / "class_to_idx.json").read_text())
    assert mapping == json.loads((FROZEN / "class_to_idx.json").read_text())
    records = read_csv(path / "manifest.csv")
    included = [r for r in records if r["status"] == "included"]
    assert len(records) == stamp["candidate_count"]
    for key in ("sha256", "pixel_d4_sha256", "output_relative_path"):
        assert len({r[key] for r in included}) == len(included)
    partitions = {}
    for row in included:
        assert file_sha256(path / row["output_relative_path"]) == row["sha256"]
        assert mapping[row["class_name"]] == int(row["class_id"])
        assert PurePosixPath(row["output_relative_path"]).parts[:2] == (row["split"], row["class_name"])
        assert row["split"] == partitions.setdefault(row["group_id"], row["split"])
        assert row["split"] in {"train", "val", "test"}
        assert (row["source"] == "huang") == (row["split"] == "test")
    assert included and any(r["split"] == "test" for r in included)
    assert any(r["split"] == "train" for r in included)
    print(f"Verified {len(included):,} image hashes, class IDs and grouped splits", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fetch", "build", "verify"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tomato-only", action="store_true")
    args = parser.parse_args()
    if args.operation == "fetch":
        fetch(args.tomato_only)
    elif not args.output:
        parser.error("--output is required for build and verify")
    elif args.operation == "build":
        build(args.output, args.tomato_only)
    else:
        verify(args.output)
