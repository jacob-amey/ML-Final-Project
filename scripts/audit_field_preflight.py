"""Restore pinned publisher samples and audit them before considering a split."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from prepare_blight_dataset import Photo, deduplicate, filter_protected, inspect, protected_hashes
from build_pakistan_tomato_development import protection as protect_previous
from prepare_okara_holdout import extend_protected
from prepare_final_dataset import file_sha256, read_csv, write_csv
from prepare_tom2024 import check_bytes, fetch_registered_images, local_path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/leafcheck_field_v9/field-preflight"
MANIFESTS = ROOT / "data/manifests"
PIN = MANIFESTS / "leafcheck_field_v9_preflight_files.csv"
IIT = MANIFESTS / "leafcheck_iit_potato_intake_files.csv"


def protected_field_sources():
    protection = protect_previous()
    for name in (
        "leafcheck_pakistan_tomato_dev_v1", "leafcheck_plantcity_holdout_v1",
        "leafcheck_ccmt_crop_v1", "leafcheck_crop_confirmation_v1",
        "leafcheck_imagesafari_crop_v1", "leafcheck_hossain_tomato_v1",
    ):
        protection = extend_protected(protection, ROOT / "data/processed" / name,
                                      MANIFESTS / f"{name}_release.json")
    return protection


def audit(output, registry, raw):
    records = read_csv(registry)
    expected = 305 if registry == IIT else 72
    if len(records) != expected or len({row["file_id"] for row in records}) != len(records):
        raise ValueError("Preflight membership changed")

    def collect(row):
        content = local_path(row, raw).read_bytes()
        check_bytes(content, row)
        source = row.get("source", "iit_potato")
        photo = Photo(source, row["filename"], row["url"], row["source_label"], row["source_label"],
                      "candidate", source, "", content)
        inspect(photo)
        photo.content = b""
        photo.device = photo.device.rstrip("\0")
        return photo

    with ThreadPoolExecutor(6) as pool:
        photos = list(pool.map(collect, records))
    protection = protected_hashes() if registry != IIT else protected_field_sources()
    filter_protected(photos, protection)
    for source in sorted({photo.source for photo in photos}):
        deduplicate([photo for photo in photos if photo.source == source])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "audit.csv"
    write_csv(path, [{
        **row, "screen": photo.reason or "no_candidate", "related_file": photo.related,
        "width": photo.width, "height": photo.height, "pixel_d4_sha256": photo.pixel_hash,
        "exif_date": photo.date, "exif_device": photo.device,
    } for row, photo in zip(records, photos, strict=True)])
    details = {}
    for source, label in sorted({(photo.source, photo.class_name) for photo in photos}):
        candidates = [p for p in photos if p.source == source and p.class_name == label]
        details[f"{source}/{label}"] = {
            "sample_count": len(candidates),
            "screens": Counter(p.reason or "no_candidate" for p in candidates),
            "exif_days": Counter(p.date.split()[0] for p in candidates if p.date),
            "exif_devices": Counter(p.device for p in candidates if p.device),
        }
    result = {
        "source_files_sha256": file_sha256(registry), "audit_sha256": file_sha256(path),
        "master_fingerprint": protection[0], "preflight": details,
        "scope": "Exact raw bytes, orientation-invariant decoded bytes and perceptual screening against master and prior releases, plus within-source grouping."
                 if registry == IIT else
                 "Master and PlantDoc raw bytes, orientation-invariant decoded bytes and perceptual screening of the published sample, plus within-source grouping.",
        "split_assigned": False, "photos_admitted": 0, "inference_run": False,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "audit"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iit-potato", action="store_true")
    args = parser.parse_args()
    registry = IIT if args.iit_potato else PIN
    raw = RAW.parent / "iit-potato" if args.iit_potato else RAW
    output = args.output or ROOT / "runs/field_v9" / ("iit-potato" if args.iit_potato else "preflight")
    if args.action == "fetch":
        fetch_registered_images(registry, raw)
    else:
        audit(output, registry, raw)
