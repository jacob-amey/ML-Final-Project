"""Screen the pinned Okara potato collection without running any model."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from prepare_blight_dataset import FROZEN, MANIFESTS, ROOT, Photo, deduplicate, filter_protected, inspect
from prepare_field_expansion import protected_with_field
from prepare_final_dataset import file_sha256, perceptual_hashes, read_csv, write_csv

PIN = MANIFESTS / "leafcheck_okara_v1_source.json"
V2 = ROOT / "data/processed/leafcheck_field_v2_sampled"
ARCHIVE = ROOT / "data/raw/leafcheck_field_v3/okara.zip"
DEFAULT = ROOT / "data/processed/leafcheck_okara_v1"


def fetch():
    source = json.loads(PIN.read_text())
    if not ARCHIVE.exists():
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        temporary = ARCHIVE.with_suffix(".download")
        subprocess.run(["curl", "-fLSs", "--max-time", "180", source["archive_url"], "-o", str(temporary)],
                       check=True)
        if file_sha256(temporary) != source["archive_sha256"]:
            raise ValueError("The mirror bytes changed")
        temporary.replace(ARCHIVE)
    if file_sha256(ARCHIVE) != source["archive_sha256"]:
        raise ValueError("The local archive does not match the registered mirror")


def protected_with_v2():
    fingerprint, count, external, raw, exact, locators, visuals, offsets = protected_with_field()
    metadata = json.loads((MANIFESTS / "leafcheck_field_v2_sampled_release.json").read_text())
    assert all(file_sha256(V2 / name) == value for name, value in metadata["files_sha256"].items())
    additions, new_offsets = [], []
    for row in read_csv(V2 / "manifest.csv"):
        if row["status"] != "included":
            continue
        path = V2 / row["output_relative_path"]
        locator = "leafcheck_field_v2_sampled:" + row["image_id"]
        assert file_sha256(path) == row["sha256"]
        raw.setdefault(row["sha256"], locator)
        exact.setdefault(row["pixel_d4_sha256"], locator)
        with Image.open(path) as original:
            visual = perceptual_hashes(ImageOps.exif_transpose(original).convert("RGB"))
        additions.extend(visual)
        new_offsets.extend([len(locators)] * len(visual))
        locators.append(locator)
    return (fingerprint, count, external, raw, exact, locators,
            np.concatenate((visuals, np.array(additions, np.uint64))),
            np.concatenate((offsets, np.array(new_offsets, np.int64))))


def ensure_entry(row, labels, output):
    if row["status"] != "included":
        assert row["reason"] and not row["output_relative_path"]
        return
    relative = PurePosixPath(row["output_relative_path"])
    assert relative.parts[:2] == ("test", row["class_name"])
    assert labels[row["class_name"]] == int(row["class_id"])
    assert row["image_id"] == "sha256:" + row["sha256"]
    assert file_sha256(output / relative) == row["sha256"]


def verify(output):
    release = json.loads((output / "release.json").read_text())
    rows = read_csv(output / "manifest.csv")
    mapping = json.loads((FROZEN / "class_to_idx.json").read_text())
    for name, expected in release["files_sha256"].items():
        assert file_sha256(output / name) == expected
    assert len(rows) == release["candidate_count"]
    for row in rows:
        ensure_entry(row, mapping, output)
    included = [row for row in rows if row["status"] == "included"]
    assert Counter(row["class_name"] for row in included) == release["class_counts"]
    for key in ("image_id", "pixel_d4_sha256"):
        assert len({row[key] for row in included}) == len(included)
    assert len(release["class_counts"]) == 3 and min(release["class_counts"].values()) >= 50
    print(f"Verified {len(included)} test-only files", flush=True)


def build(output):
    if output.exists():
        raise FileExistsError(output)
    fetch()
    source = json.loads(PIN.read_text())
    classes = source["label_map"]
    mapping = json.loads((FROZEN / "class_to_idx.json").read_text())
    candidates = []
    with ZipFile(ARCHIVE) as archive:
        for name in sorted(archive.namelist()):
            parts = PurePosixPath(name).parts
            if (len(parts) == 4 and parts[0] == "PLD_3_Classes_256" and parts[1] in {"Training", "Validation", "Testing"}
                    and parts[2] in classes and parts[3].lower().endswith(".jpg")):
                candidates.append(Photo("okara", name, source["archive_url"], classes[parts[2]], parts[2],
                                        "test", "okara:full_collection", "", archive.read(name)))
    assert len(candidates) == 4072
    with ThreadPoolExecutor(6) as pool:
        list(pool.map(inspect, candidates))
    overlap = protected_with_v2()
    filter_protected(candidates, overlap)
    deduplicate(candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".okara-") as work:
        stage = Path(work) / "release"
        stage.mkdir()
        rows = []
        for photo in candidates:
            if not photo.reason:
                photo.output = f"test/{photo.class_name}/{photo.sha256}{photo.extension}"
                path = stage / photo.output
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(photo.content)
            rows.append(dict(image_id="sha256:" + photo.sha256, source="okara", class_name=photo.class_name,
                             class_id=mapping[photo.class_name], source_label=photo.source_label,
                             split="" if photo.reason else "test", group_id=photo.group, source_locator=photo.locator,
                             status="excluded" if photo.reason else "included", reason=photo.reason,
                             related_to=photo.related, sha256=photo.sha256, pixel_d4_sha256=photo.pixel_hash,
                             output_relative_path=photo.output))
        write_csv(stage / "manifest.csv", rows)
        (stage / "source.json").write_bytes(PIN.read_bytes())
        release = {
            "source": source["version"], "candidate_count": len(candidates),
            "master_fingerprint": overlap[0], "master_records_screened": overlap[1],
            "plantdoc_records_screened": overlap[2], "roles": "test only",
            "files_sha256": {name: file_sha256(stage / name) for name in ("manifest.csv", "source.json")},
            "class_counts": dict(Counter(p.class_name for p in candidates if not p.reason)),
            "exclusions": dict(Counter(p.reason for p in candidates if p.reason)),
            "specimen_ids_available": False,
        }
        (stage / "release.json").write_text(json.dumps(release, indent=2) + "\n")
        verify(stage)
        stage.rename(output)
    print(json.dumps(release, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "build", "verify"))
    parser.add_argument("--output", default=DEFAULT, type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch()
    elif args.action == "build":
        build(args.output)
    else:
        verify(args.output)
