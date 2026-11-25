"""Sample metadata-matched Image Safari crop images into site-disjoint development splits."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

from prepare_ccmt_crop_training import OUTPUT as CCMT
from prepare_ccmt_crop_training import MANIFESTS, ROOT, protection
from prepare_crop_confirmation import RELEASE as CONFIRMATION
from prepare_final_dataset import file_sha256, read_csv, write_csv
from prepare_plantcity_holdout import DEFAULT as PLANTCITY, PAKISTAN
from prepare_tom2024 import Photo, check_bytes, deduplicate, extend_protected, filter_protected, inspect


NAME = "leafcheck_imagesafari_crop_v1"
INTAKE = MANIFESTS / "leafcheck_imagesafari_intake.json"
RAW = ROOT / "data/raw/leafcheck_field_v7/imagesafari"
OUTPUT_DIR = ROOT / "data/processed" / NAME
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def request(url, limit):
    with urlopen(url, timeout=30) as response:
        return response.read(limit + 1)


def path_url(source, key):
    return source["bucket"] + quote(key)


def listed(source, prefix, quota, xpath):
    query = urlencode({"list-type": 2, "prefix": prefix, "delimiter": "/", "max-keys": quota})
    content = request(source["bucket"] + "?" + query, 128_000)
    root = ElementTree.fromstring(content)
    return [entry.text for entry in root.findall(xpath, NS)]


def collect_group(group, source, raw):
    folder, expected, crop, site, split = group
    prefix = f"ImageSafari/{folder}/metadata/{site}/"
    dates = listed(source, prefix, source["max_site_days_listed"], "s:CommonPrefixes/s:Prefix")
    valid_days = []
    for key in dates:
        day = key.removesuffix("/").rsplit("/", 1)[-1]
        try:
            date.fromisoformat(day)
        except ValueError:
            continue
        valid_days.append((day, key))
    result = []
    for day, directory in sorted(valid_days)[:source["days_per_site_and_crop"]]:
        objects = listed(source, directory, source["max_sidecars_listed_per_day"], "s:Contents/s:Key")
        candidates = sorted((key for key in objects if key.endswith(".json")), key=lambda key: sha256(key.encode()).digest())
        kept = 0
        for sidecar in candidates:
            ver = request(path_url(source, sidecar), 64_000)
            meta = json.loads(ver)
            if (meta.get("type") != "original" or meta.get("synthetic_image_properties")
                    or (meta.get("agronomic_properties") or {}).get("crop_type") != expected):
                continue
            key = sidecar.replace("/metadata/", "/images/").removesuffix(".json") + ".jpg"
            try:
                buf = request(path_url(source, key), 12_582_912)
            except HTTPError as exc:
                if exc.code == 404:
                    continue
                raise
            if len(buf) > 12_582_912:
                continue
            digest = sha256(buf).hexdigest()
            (raw / (digest + ".jpg")).write_bytes(buf)
            result.append({"crop": crop, "split": split, "site": site, "day": day, "key": key, "sidecar": sidecar,
                           "sidecar_sha256": sha256(ver).hexdigest(), "sha256": digest, "size": len(buf)})
            kept += 1
            if kept == source["views_per_day"]:
                break
        print(site, crop, day, kept, flush=True)
    return result


def acquire(source, raw, registry):
    if registry.exists():
        raise FileExistsError(registry)
    with ThreadPoolExecutor(6) as executor:
        records = [row for group in executor.map(lambda g: collect_group(g, source, raw), source["groups"]) for row in group]
    if len(records) != len({row["key"] for row in records}):
        raise ValueError("Public object keys occurred more than once")
    write_csv(registry.with_suffix(".tmp"), records)
    registry.with_suffix(".tmp").replace(registry)
    print("Pinned Image Safari objects", len(records), flush=True)


def build(source, raw, registry, output):
    if output.exists():
        raise FileExistsError(output)
    allowed = {(crop, site): split for _, _, crop, site, split in source["groups"]}
    groups = {}
    for (_, site), split in allowed.items():
        if groups.setdefault(site, split) != split:
            raise ValueError("A complete site cannot span train and validation")
    records = read_csv(registry)

    def prepare(row):
        content = (raw / (row["sha256"] + ".jpg")).read_bytes()
        check_bytes(content, row)
        if allowed[row["crop"], row["site"]] != row["split"]:
            raise ValueError("Capture site was reassigned")
        p = Photo("imagesafari", row["key"], path_url(source, row["key"]), row["crop"], "", row["split"],
                  f"imagesafari:{row['site']}", "", content)
        inspect(p)
        return p

    with ThreadPoolExecutor(6) as executor:
        photos = list(executor.map(prepare, records))
    known = protection()
    for release in (PAKISTAN, PLANTCITY, CCMT, CONFIRMATION):
        known = extend_protected(known, release, MANIFESTS / f"{release.name}_release.json")
    filter_protected(photos, known)
    deduplicate(photos)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".imagesafari-") as tmp:
        directory = Path(tmp) / "release"
        directory.mkdir()
        final = []
        for row, photo in zip(records, photos, strict=True):
            if not photo.reason:
                photo.output = f"{photo.split}/{photo.class_name}/{photo.sha256}{photo.extension}"
                target = directory / photo.output
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(photo.content)
            final.append({**row, "image_id": "sha256:" + photo.sha256, "group_id": photo.group,
                          "status": "included" if not photo.reason else "excluded",
                          "reason": photo.reason, "related_to": photo.related, "pixel_d4_sha256": photo.pixel_hash,
                          "output_relative_path": photo.output})
        write_csv(directory / "manifest.csv", final)
        (directory / "source_files.csv").write_bytes(registry.read_bytes())
        included = [r for r in final if r["status"] == "included"]
        assert all(file_sha256(directory / r["output_relative_path"]) == r["sha256"] for r in included)
        assert len({r["pixel_d4_sha256"] for r in included}) == len(included)
        files = {name: file_sha256(directory / name) for name in ("manifest.csv", "source_files.csv")}
        report = {
            "version": NAME, "role": source["role"], "intake_sha256": file_sha256(INTAKE),
            "base_release": known[0], "files_sha256": files,
            "fingerprint": sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "candidates": len(photos), "split_crops": {split: dict(Counter(
                photo.class_name for photo in photos if not photo.reason and photo.split == split
            )) for split in ("train", "val")},
            "excluded": dict(Counter(photo.reason for photo in photos if photo.reason)),
        }
        (directory / "release.json").write_text(json.dumps(report, indent=2) + "\n")
        directory.rename(output)
    print(json.dumps(report, indent=2), flush=True)


def restore(source, raw, registry):
    def save(row):
        path = raw / (row["sha256"] + ".jpg")
        contents = path.read_bytes() if path.exists() else request(path_url(source, row["key"]), 12_582_912)
        check_bytes(contents, row)
        if not path.exists():
            path.write_bytes(contents)
    with ThreadPoolExecutor(6) as executor:
        for _ in executor.map(save, read_csv(registry)):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "restore", "build"))
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--registry", type=Path, default=MANIFESTS / f"{NAME}_files.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    source = json.loads(INTAKE.read_text())
    with (args.raw / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "acquire":
            acquire(source, args.raw, args.registry)
        elif args.action == "restore":
            restore(source, args.raw, args.registry)
        else:
            build(source, args.raw, args.registry, args.output)
