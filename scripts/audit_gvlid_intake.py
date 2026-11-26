"""Audit the pinned GVLiD archive before assigning any data or evaluation role."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from prepare_blight_dataset import Photo, deduplicate, filter_protected, inspect, protected_hashes
from prepare_final_dataset import file_sha256, write_csv


ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "data/manifests/leafcheck_gvlid_intake.json"


def audit(output):
    pin = json.loads(PIN.read_text())
    archive = ROOT / pin["path"]
    if archive.stat().st_size != pin["bytes"] or file_sha256(archive) != pin["sha256"]:
        raise ValueError("GVLiD archive does not match the publisher record")
    source = archive.parent / "original/DATASET"
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    by_name = {row["image_filename"]: row for row in metadata}
    files = sorted(path for path in source.glob("*/*") if path.suffix.lower() == ".jpg")
    filenames = {path.name for path in files}
    if len(files) != pin["expected_jpeg_count"] or len(filenames) != len(files):
        raise ValueError("Unexpected GVLiD image count or nonunique filename")
    if len(by_name) != len(metadata) or len({row["Image_ID"] for row in metadata}) != len(metadata):
        raise ValueError("GVLiD image metadata identities are not unique")

    def photograph(path):
        data = by_name.get(path.name, {})
        # Keep labels as publisher strings until class compatibility review.
        photo = Photo("gvlid", str(path.relative_to(source)), pin["publisher"],
                      path.parent.name, data.get("Disease_Type", ""), "candidate",
                      "unverified_gvlid_capture", "", path.read_bytes())
        inspect(photo)
        photo.content = b""
        return photo

    with ThreadPoolExecutor(6) as pool:
        photos = list(pool.map(photograph, files))
    protected = protected_hashes()
    filter_protected(photos, protected)
    deduplicate(photos)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "audit.csv"
    write_csv(path, [{
        "member": photo.locator, "publisher_class": photo.class_name, "metadata_class": photo.source_label,
        "sha256": photo.sha256, "pixel_d4_sha256": photo.pixel_hash, "screen": photo.reason or "no_candidate",
        "match": photo.related, "exif_date": photo.date, "exif_device": photo.device,
    } for photo in photos])
    screens = defaultdict(Counter)
    for photo in photos:
        screens[photo.class_name][photo.reason or "no_candidate"] += 1
    dates = Counter(row["Environmental_Conditions"]["Time_of_Capture"][:10] for row in metadata)
    metadata_label_check = defaultdict(Counter)
    classes_per_pixel_hash = defaultdict(set)
    for photo in photos:
        metadata_label_check[photo.class_name][photo.source_label] += 1
        if photo.pixel_hash:
            classes_per_pixel_hash[photo.pixel_hash].add(photo.class_name)
    result = {
        "archive_sha256": pin["sha256"], "metadata_sha256": file_sha256(metadata_path),
        "audit_csv_sha256": file_sha256(path), "master_fingerprint": protected[0],
        "photos": len(photos), "missing_metadata": len(filenames - by_name.keys()),
        "extra_metadata": len(by_name.keys() - filenames), "publisher_metadata_dates": dates,
        "publisher_metadata_vineyards": len({row["Location"]["Vineyard"] for row in metadata}),
        "publisher_metadata_devices": Counter(row["Device_Specifications"]["Device_Name"] for row in metadata),
        "exif_date_count": sum(bool(photo.date) for photo in photos),
        "exif_days": Counter(photo.date.split()[0] for photo in photos if photo.date),
        "exif_devices": Counter(photo.device for photo in photos if photo.device),
        "by_source_class": screens, "image_folder_vs_metadata": metadata_label_check,
        "folder_metadata_disagreements": sum(
            photo.class_name.casefold() != photo.source_label.casefold() for photo in photos
        ),
        "exact_decoded_groups_with_conflicting_folders": sum(len(labels) > 1 for labels in classes_per_pixel_hash.values()),
        "scope": "Master and PlantDoc raw-byte, decoded D4 and perceptual screening; within-source duplicate grouping. Visual candidates are heuristic.",
        "split_assigned": False, "photos_admitted": 0, "inference_run": False,
    }
    (output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
