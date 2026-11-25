"""Restore pinned WASCAL training images and keep other data roles intact."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from prepare_blight_dataset import FROZEN, MANIFESTS, ROOT, Photo, deduplicate, filter_protected, inspect
from prepare_final_dataset import file_sha256, read_csv, write_csv
from prepare_okara_holdout import DEFAULT as OKARA
from prepare_okara_holdout import extend_protected, protected_with_v2

NAME = "leafcheck_tom2024_v1"
REGISTRY = MANIFESTS / f"{NAME}_files.csv"
SOURCE = MANIFESTS / f"{NAME}_source.json"
RAW = ROOT / "data/raw/leafcheck_field_v3/tom2024"
DEFAULT = ROOT / "data/processed" / NAME


def local_path(row, raw=RAW):
    return raw / (row["file_id"] + ".bin")


def check_bytes(data, row):
    if len(data) != int(row["size"]) or hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"Publisher bytes changed for {row['file_id']}")


def fetch_one(row, raw):
    path = local_path(row, raw)
    if path.exists():
        check_bytes(path.read_bytes(), row)
        return
    stage = path.with_suffix(".tmp")
    subprocess.run(["curl", "-fLSs", "--retry", "2", "--max-time", "45",
                    row["url"], "-o", str(stage)], check=True)
    check_bytes(stage.read_bytes(), row)
    stage.replace(path)


def fetch_registered_images(registry, raw):
    raw.mkdir(parents=True, exist_ok=True)
    records = read_csv(registry)
    with ThreadPoolExecutor(4) as pool:
        for count, _ in enumerate(pool.map(lambda row: fetch_one(row, raw), records), 1):
            if count % 50 == 0 or count == len(records):
                print(f"Verified {count}/{len(records)} publisher images", flush=True)


def fetch():
    fetch_registered_images(REGISTRY, RAW)


def build_registered_release(output, registry, source_path, raw, source_id, split, protection, prepare_candidate=None):
    if output.exists():
        raise FileExistsError(output)
    rows = read_csv(registry)
    source = json.loads(source_path.read_text())
    mapping = json.loads((FROZEN / "class_to_idx.json").read_text())

    def photo(row):
        content = local_path(row, raw).read_bytes()
        check_bytes(content, row)
        p = Photo(source_id, f"{row['source_folder']}/{row['filename']}:{row['file_id']}", row["url"],
                  row["class_name"], row["source_folder"].rsplit("/", 1)[-1], split,
                  f"{source_id}:collection", "", content)
        inspect(p)
        return p

    with ThreadPoolExecutor(6) as pool:
        candidates = list(pool.map(prepare_candidate or photo, rows))
    protected = protection()
    filter_protected(candidates, protected)
    deduplicate(candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".tom2024-") as directory:
        staging = Path(directory) / "release"
        staging.mkdir()
        records = []
        for p in candidates:
            if not p.reason:
                p.output = f"{split}/{p.class_name}/{p.sha256}{p.extension}"
                path = staging / p.output
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(p.content)
            records.append(dict(image_id="sha256:" + p.sha256, source=p.source, class_name=p.class_name,
                                source_label=p.source_label, class_id=mapping[p.class_name], group_id=p.group,
                                split="" if p.reason else split, status="excluded" if p.reason else "included",
                                reason=p.reason, related_to=p.related, source_locator=p.locator, source_url=p.url,
                                width=p.width, height=p.height, sha256=p.sha256, pixel_d4_sha256=p.pixel_hash,
                                output_relative_path=p.output))
        write_csv(staging / "manifest.csv", records)
        (staging / "source.json").write_text(json.dumps(source, indent=2) + "\n")
        (staging / "source_files.csv").write_bytes(registry.read_bytes())
        seal = {"version": source["version"], "base_release": protected[0], "candidates": len(candidates),
                "files_sha256": {name: file_sha256(staging / name) for name in (
                    "manifest.csv", "source.json", "source_files.csv")},
                f"{split}_only_counts": dict(Counter(p.class_name for p in candidates if not p.reason)),
                "exclusions": dict(Counter(p.reason for p in candidates if p.reason))}
        seal["fingerprint"] = hashlib.sha256(json.dumps(seal["files_sha256"], sort_keys=True).encode()).hexdigest()
        (staging / "release.json").write_text(json.dumps(seal, indent=2) + "\n")
        included = [r for r in records if r["status"] == "included"]
        assert len({r["output_relative_path"] for r in included}) == len(included)
        assert all(r["split"] == split and file_sha256(staging / r["output_relative_path"]) == r["sha256"]
                   for r in included)
        staging.rename(output)
    print(json.dumps(seal, indent=2), flush=True)


def protected_with_okara():
    return extend_protected(protected_with_v2(), OKARA, MANIFESTS / "leafcheck_okara_v1_release.json")


def build(output):
    build_registered_release(output, REGISTRY, SOURCE, RAW, "tom2024", "train", protected_with_okara)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "build"))
    parser.add_argument("--output", default=DEFAULT, type=Path)
    args = parser.parse_args()
    if args.action == "fetch":
        fetch()
    else:
        build(args.output)
