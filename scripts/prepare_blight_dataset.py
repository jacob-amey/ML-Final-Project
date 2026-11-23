"""Build a versioned tomato and potato blight supplement from pinned public photos."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.request import Request, urlopen
import warnings
import zipfile

import numpy as np
from PIL import Image, ImageOps

from prepare_final_dataset import canonical_pixel_hash, file_sha256, perceptual_hashes, read_csv, write_csv

ROOT = Path(__file__).resolve().parents[1]
NAME = "leafcheck_blight_v1"
MANIFESTS = ROOT / "data/manifests"
SOURCES = MANIFESTS / f"{NAME}_sources.json"
BANU_ITEMS = MANIFESTS / f"{NAME}_sfldata.csv"
FIELD_ITEMS = MANIFESTS / f"{NAME}_field_photos.csv"
RAW = ROOT / "data/raw" / NAME
RELEASE = ROOT / "data/processed" / NAME
FROZEN = ROOT / "data/processed/leafcheck177_final"
PLANTDOC = MANIFESTS / "leafcheck177_external.csv"
SOURCE = json.loads(SOURCES.read_text())
DISTANCE = SOURCE["near_duplicate_distance"]
EXTENSIONS = {"JPEG": ".jpg", "PNG": ".png"}


@dataclass
class Photo:
    source: str
    locator: str
    url: str
    class_name: str
    source_label: str
    split: str
    group: str
    object_id: str
    content: bytes
    date: str = ""
    device: str = ""
    background: str = ""
    sha256: str = ""
    pixel_hash: str = ""
    perceptual: list[int] = field(default_factory=list)
    width: int = 0
    height: int = 0
    extension: str = ""
    reason: str = ""
    related: str = ""
    output: str = ""
    source_page: str = ""
    credit: str = ""
    license: str = ""
    license_url: str = ""


def inputs():
    for source in SOURCE["sources"].values():
        for item in source["files"]:
            yield RAW / item["name"], item["url"], item["sha256"]
    for row in read_csv(BANU_ITEMS):
        yield RAW / "sfld" / f"{row['drive_id']}.jpg", row["download_url"], row["sha256"]
    for row in read_csv(FIELD_ITEMS):
        yield RAW / "field" / f"{row['asset_id']}.jpg", row["download_url"], row["sha256"]


def fetch():
    """Restore missing approved originals to the local source cache."""
    for path, url, expected in inputs():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            request = Request(url, headers={"User-Agent": "LeafCheck-CSC427-dataset/1.0"})
            with tempfile.TemporaryDirectory(dir=path.parent) as directory:
                temporary = Path(directory) / path.name
                with urlopen(request, timeout=90) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                if file_sha256(temporary) != expected:
                    raise ValueError(f"Publisher bytes differ for {path.name}")
                temporary.replace(path)
            print("Fetched", path.name, flush=True)
        elif file_sha256(path) != expected:
            raise ValueError(f"Source copy differs from its pin for {path.name}")


def photos():
    sheikh_url = SOURCE["sources"]["sheikh"]["files"][0]["url"]
    with zipfile.ZipFile(RAW / "sheikh_resized.zip") as archive:
        for row in read_csv(RAW / "sheikh_metadata.csv"):
            label = row["Class_Label"]
            class_name = SOURCE["class_map"].get(label.casefold())
            if class_name is None:
                continue
            date = row["Capture_Date"]
            assert date and row["Object_ID"] and row["Augmented"] == "No"
            name = f"Resize Dataset/{label}/{row['File_Name']}"
            split = "val" if date in SOURCE["sheikh_validation_dates"] else "train"
            yield Photo("sheikh", name, sheikh_url, class_name, label, split,
                        f"sheikh:date:{date}", f"sheikh:{row['Object_ID']}", archive.read(name),
                        date, row["Camera_Device"], row["Background"])

    mtali_url = SOURCE["sources"]["mtali"]["files"][0]["url"]
    with zipfile.ZipFile(RAW / "mtali_late6.zip") as archive:
        for name in sorted(archive.namelist()):
            if not re.fullmatch(r"LATEBLT_6/Lateblt\d+\.jpg", name, re.IGNORECASE):
                continue
            yield Photo("mtali", name, mtali_url, "potato__late_blight", "Late Blight", "train",
                        "mtali:LATEBLT_6", "", archive.read(name))

    for row in read_csv(BANU_ITEMS):
        assert re.fullmatch(r"IMG_\d+\.jpe?g", row["filename"], re.IGNORECASE)
        filename = f"{row['drive_id']}.jpg"
        class_name = row["class_name"]
        yield Photo("banu", f"{row['folder_url']}/{row['filename']}", row["download_url"],
                    class_name, "Healthy" if class_name.endswith("__healthy") else "Late Blight",
                    "val", "banu:original_collection", "", (RAW / "sfld" / filename).read_bytes())

    for row in read_csv(FIELD_ITEMS):
        filename = f"{row['asset_id']}.jpg"
        photo = Photo("bugwood", filename, row["download_url"], row["class_name"],
                      row["source_label"], "val", row["group_id"], "",
                      (RAW / "field" / filename).read_bytes(),
                      background="Field per original catalog", source_page=row["source_page"],
                      credit=row["credit"], license=row["license"], license_url=row["license_url"])
        photo.reason = row["exclusion_reason"]
        yield photo


def inspect(photo):
    photo.sha256 = hashlib.sha256(photo.content).hexdigest()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(photo.content)) as original:
                photo.extension = EXTENSIONS[original.format]
                if getattr(original, "n_frames", 1) != 1:
                    photo.reason = "multiple_frames"
                    return
                exif = original.getexif()
                if not photo.date:
                    photo.date = str(exif.get(306, ""))
                    photo.device = str(exif.get(272, ""))
                # Photos from the same recorded capture second may be derived views.
                if photo.source == "mtali" and photo.date:
                    photo.object_id = f"mtali:timestamp:{photo.date}"
                rgb = ImageOps.exif_transpose(original).convert("RGB")
        photo.width, photo.height = rgb.size
        if min(rgb.size) < 32 or all(low == high for low, high in rgb.getextrema()):
            photo.reason = "too_small_or_uniform"
            return
        photo.pixel_hash = canonical_pixel_hash(rgb)
        photo.perceptual = perceptual_hashes(rgb)
    except (OSError, KeyError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        photo.reason = "invalid_or_unsupported_image"


def protected_hashes():
    locked = json.loads((MANIFESTS / "leafcheck177_frozen.json").read_text())
    for filename, digest in locked["files"].items():
        assert file_sha256(FROZEN / filename) == digest
    frozen_info = json.loads((FROZEN / "dataset_info.json").read_text())
    raw, exact, paths, phashes, indexes = {}, {}, [], [], []

    def add(locator, byte_hash, pixel_hash, visual):
        raw.setdefault(byte_hash, locator)
        if pixel_hash:
            exact.setdefault(pixel_hash, locator)
        if visual:
            indexes.extend([len(paths)] * len(visual))
            phashes.extend(visual)
            paths.append(locator)

    with (FROZEN / "metadata/image_audit.jsonl").open() as stream:
        count = 0
        for line in stream:
            row = json.loads(line)
            add(f"master:{row['master_path']}", row.get("actual_sha256"),
                row.get("pixel_d4_sha256"), row.get("phashes"))
            count += 1
    assert count == frozen_info["original_images"]

    # PlantDoc is a separate previously evaluated source, outside the master.
    external = [row for row in read_csv(PLANTDOC) if row["status"] == "included"]
    assert len(external) == 183
    for row in external:
        path = ROOT / row["image_path"]
        assert file_sha256(path) == row["sha256"]
        with Image.open(path) as original:
            rgb = ImageOps.exif_transpose(original).convert("RGB")
        add(f"plantdoc:{row['image_id']}", row["sha256"], canonical_pixel_hash(rgb), perceptual_hashes(rgb))

    return locked["fingerprint"], count, len(external), raw, exact, paths, np.array(phashes, dtype=np.uint64), np.array(indexes)


def filter_protected(candidates, protected):
    _, _, _, raw, exact, paths, phashes, indexes = protected
    for photo in candidates:
        if photo.reason:
            continue
        if photo.sha256 in raw:
            photo.reason, photo.related = "existing_raw_match", raw[photo.sha256]
        elif photo.pixel_hash in exact:
            photo.reason, photo.related = "existing_pixel_match", exact[photo.pixel_hash]
        else:
            differences = np.bitwise_count(phashes ^ np.uint64(photo.perceptual[0]))
            closest = int(differences.argmin())
            if differences[closest] <= DISTANCE:
                photo.reason, photo.related = "existing_visual_candidate", paths[indexes[closest]]


def deduplicate(candidates):
    """Keep one view per same-label cluster and quarantine split or label conflicts."""
    active = [photo for photo in candidates if not photo.reason]
    parent = list(range(len(active)))

    def root(index):
        while index != parent[index]:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    phashes, indexes = [], []
    identities, captures = {}, {}
    for index, photo in enumerate(active):
        for key, lookup in [(photo.pixel_hash, identities), (photo.object_id, captures)]:
            if key:
                other = lookup.setdefault(key, index)
                parent[root(index)] = root(other)
        phashes.extend(photo.perceptual)
        indexes.extend([index] * len(photo.perceptual))

    indexed_hashes = np.array(phashes, dtype=np.uint64)
    indexed_photos = np.array(indexes, dtype=np.int32)
    for index, photo in enumerate(active):
        matches = np.bitwise_count(indexed_hashes ^ np.uint64(photo.perceptual[0])) <= DISTANCE
        for other in np.unique(indexed_photos[matches]):
            parent[root(index)] = root(int(other))

    clusters = defaultdict(list)
    for index, photo in enumerate(active):
        clusters[root(index)].append(photo)
    for members in clusters.values():
        if len(members) == 1:
            continue
        classes, splits = {photo.class_name for photo in members}, {photo.split for photo in members}
        keeper = min(members, key=lambda p: (-p.width * p.height, p.source, p.locator))
        for photo in members:
            if len(classes) > 1:
                photo.reason = "new_cluster_label_conflict"
            elif len(splits) > 1:
                photo.reason = "new_cluster_cross_split"
            elif photo is not keeper:
                photo.reason = "new_cluster_repeated_view"
            if photo.reason:
                witness = keeper if photo is not keeper else next(other for other in members if other is not photo)
                photo.related = f"{witness.source}:{witness.locator}"


def record(photo, mapping):
    source = SOURCE["sources"][photo.source]
    return {
        "image_id": f"sha256:{photo.sha256}",
        "class_id": mapping[photo.class_name],
        "class_name": photo.class_name,
        "split": photo.split if not photo.reason else "",
        "planned_split": photo.split,
        "status": "excluded" if photo.reason else "included",
        "reason": photo.reason,
        "related_to": photo.related,
        "group_id": photo.group,
        "object_id": photo.object_id,
        "source": photo.source,
        "source_label": photo.source_label,
        "source_locator": photo.locator,
        "source_page": photo.source_page or source["page"],
        "source_file_url": photo.url,
        "credit": photo.credit or source["credit"],
        "license": photo.license or SOURCE["license"],
        "license_url": photo.license_url or SOURCE["license_url"],
        "capture_date": photo.date,
        "camera": photo.device,
        "background": photo.background,
        "width": photo.width,
        "height": photo.height,
        "sha256": photo.sha256,
        "pixel_d4_sha256": photo.pixel_hash,
        "output_relative_path": photo.output,
    }


def build(output):
    if output.exists():
        raise FileExistsError(f"Preserving existing release at {output}")
    for path, _, expected in inputs():
        assert file_sha256(path) == expected, f"Changed source: {path}"
    mapping = json.loads((FROZEN / "class_to_idx.json").read_text())
    assert len(mapping) == 177

    candidates = list(photos())
    for photo in candidates:
        inspect(photo)
    print(f"Decoded and fingerprinted {len(candidates)} candidates", flush=True)
    protected = protected_hashes()
    filter_protected(candidates, protected)
    deduplicate(candidates)
    summary = Counter((p.split, p.class_name) for p in candidates if not p.reason)
    for class_name in set(SOURCE["class_map"].values()) | {"potato__late_blight"}:
        assert all(summary[(split, class_name)] for split in ("train", "val")), class_name

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{NAME}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        for photo in candidates:
            if not photo.reason:
                photo.output = f"{photo.split}/{photo.class_name}/{photo.sha256}{photo.extension}"
                path = staging / photo.output
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(photo.content)

        rows = [record(photo, mapping) for photo in candidates]
        write_csv(staging / "manifest.csv", rows)
        write_csv(staging / "split_summary.csv", [
            {"split": split, "class_name": label, "images": count}
            for (split, label), count in sorted(summary.items())
        ])
        shutil.copyfile(FROZEN / "class_to_idx.json", staging / "class_to_idx.json")
        shutil.copyfile(SOURCES, staging / "sources.json")
        shutil.copyfile(BANU_ITEMS, staging / "sfld_inventory.csv")
        shutil.copyfile(FIELD_ITEMS, staging / "field_inventory.csv")
        checksum_names = ("manifest.csv", "split_summary.csv", "class_to_idx.json", "sources.json",
                          "sfld_inventory.csv", "field_inventory.csv")
        hashes = {name: file_sha256(staging / name) for name in checksum_names}
        info = {
            "version": NAME,
            "fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "files_sha256": hashes,
            "protected_release_fingerprint": protected[0],
            "master_records_screened": protected[1],
            "plantdoc_records_screened": protected[2],
            "candidates": len(rows),
            "counts": dict(Counter(row["split"] for row in rows if row["status"] == "included")),
            "exclusions": dict(sorted(Counter(row["reason"] for row in rows if row["reason"]).items())),
            "source_counts": [{"source": source, "split": split, "class_name": label, "images": count}
                for (source, split, label), count in sorted(Counter(
                    (row["source"], row["split"], row["class_name"]) for row in rows if row["status"] == "included").items())],
        }
        (staging / "release.json").write_text(json.dumps(info, indent=2) + "\n")
        verify(staging)
        if output.exists():
            raise FileExistsError(f"Another process created {output}; preserving it")
        staging.rename(output)
    print(json.dumps(info, indent=2), flush=True)


def verify(path):
    info = json.loads((path / "release.json").read_text())
    assert hashlib.sha256(json.dumps(info["files_sha256"], sort_keys=True).encode()).hexdigest() == info["fingerprint"]
    for filename, expected in info["files_sha256"].items():
        assert file_sha256(path / filename) == expected
    assert file_sha256(path / "class_to_idx.json") == file_sha256(FROZEN / "class_to_idx.json")
    mapping = json.loads((path / "class_to_idx.json").read_text())
    rows = read_csv(path / "manifest.csv")
    included = [row for row in rows if row["status"] == "included"]
    assert len(rows) == info["candidates"]
    assert Counter(row["split"] for row in included) == info["counts"]
    assert Counter((row["split"], row["class_name"]) for row in included) == {
        (row["split"], row["class_name"]): int(row["images"]) for row in read_csv(path / "split_summary.csv")
    }
    assert all(row["reason"] for row in rows if row["status"] == "excluded")
    for field_name in ("sha256", "pixel_d4_sha256", "output_relative_path"):
        assert len({row[field_name] for row in included}) == len(included)
    for field_name in ("group_id", "object_id"):
        groups = defaultdict(set)
        for row in included:
            if row[field_name]:
                groups[row[field_name]].add(row["split"])
        assert all(len(splits) == 1 for splits in groups.values())
    expected_paths = set()
    for row in included:
        filename = row["output_relative_path"]
        assert PurePosixPath(filename).parts[:2] == (row["split"], row["class_name"])
        assert row["split"] in {"train", "val"} and int(row["class_id"]) == mapping[row["class_name"]]
        assert row["image_id"] == "sha256:" + row["sha256"]
        assert file_sha256(path / filename) == row["sha256"]
        expected_paths.add(filename)
    assert expected_paths == {entry.relative_to(path).as_posix() for split in ("train", "val")
                              for entry in (path / split).rglob("*") if entry.is_file()}
    print(f"Verified {len(included):,} images, stable class IDs and split groups", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["fetch", "build", "verify"])
    parser.add_argument("--output", type=Path, default=RELEASE)
    args = parser.parse_args()
    operation = args.operation
    if operation == "fetch":
        fetch()
    elif operation == "build":
        build(args.output)
    else:
        verify(args.output)
