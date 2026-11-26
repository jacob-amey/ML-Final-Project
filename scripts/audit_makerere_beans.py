"""Screen the publisher's 2020 Uganda bean archives without assigning split roles."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from audit_field_preflight import protected_field_sources
from prepare_blight_dataset import DISTANCE, ROOT, Photo, deduplicate, filter_protected, inspect
from prepare_final_dataset import file_sha256, write_csv
from prepare_okara_holdout import extend_protected
from inspect_tanzania_bean import OUTPUT as TANZANIA_RAW


RAW = ROOT / "data/raw/leafcheck_field_v12/makerere-beans"
MANIFESTS = ROOT / "data/manifests"
PUBLISHER = "https://github.com/AI-Lab-Makerere/ibean"
REVISION = "27aa014ce09b193e1a6f58112d4a66e0eddb69c5"
BASE_URL = f"https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/{REVISION}/data"
ARCHIVES = {
    "train": (1034, "284fe8456ce20687f4367ae7ad94a64577e7f9fde2c2c6b1c74340ab5dc82715"),
    "validation": (133, "90b7aa1c26d91d9afff07a30bbc67a5ea34f1f1397f068d8675be09d7d0c602d"),
    "test": (128, "ca67b15d960d1e2fd9d23fd8498ce86818ead90755c630a43baf19ec4af09312"),
}
CLASSES = {"healthy": None, "bean_rust": "bean__rust", "angular_leaf_spot": None}


def protect_tanzania(protected):
    fingerprint, master_count, external_count, raw, exact, paths, hashes, indexes = protected
    extra_hashes, extra_indexes = [], []
    intake = json.loads((MANIFESTS / "leafcheck_tanzania_bean_preflight.json").read_text())
    for row in intake["records"]:
        content = (TANZANIA_RAW / f"{row['sha256']}.jpg").read_bytes()
        locator = "tanzania-bean:" + row["member"]
        photo = Photo("tanzania-bean", locator, intake["source"], row["label"], row["label"], "candidate",
                      "tanzania-bean", "", content)
        inspect(photo)
        if photo.sha256 != row["sha256"] or photo.reason:
            raise ValueError(f"Inconsistent Tanzania original at {locator}")
        raw.setdefault(photo.sha256, locator)
        exact.setdefault(photo.pixel_hash, locator)
        extra_indexes.extend([len(paths)] * len(photo.perceptual))
        extra_hashes.extend(photo.perceptual)
        paths.append(locator)
    return (fingerprint, master_count, external_count, raw, exact, paths,
            np.concatenate((hashes, np.array(extra_hashes, dtype=hashes.dtype))),
            np.concatenate((indexes, np.array(extra_indexes, dtype=indexes.dtype))))


def audit(output):
    photos = []
    provenance = {}
    for split, (expected, digest) in ARCHIVES.items():
        path = RAW / f"{split}.zip"
        url = f"{BASE_URL}/{path.name}?download=true"
        if file_sha256(path) != digest:
            raise ValueError(f"Changed published archive bytes at {path}")
        provenance[split] = {"download_url": url, "sha256": digest, "jpeg_count": expected}
        with ZipFile(path) as archive:
            members = sorted(member for member in archive.namelist() if member.lower().endswith(".jpg"))
            if len(members) != expected or len(set(members)) != expected:
                raise ValueError(f"Unexpected membership in {path}")
            for member in members:
                folder, label, _ = Path(member).parts
                if folder != split or label not in CLASSES:
                    raise ValueError(f"Unexpected publisher class or split for {member}")
                photo = Photo("makerere-beans", member, url, label, label, split, "makerere-beans", "", archive.read(member))
                inspect(photo)
                photo.content = b""
                photos.append(photo)

    protected = protected_field_sources()
    coffee = "leafcheck_decafia_v2"
    protected = extend_protected(protected, ROOT / "data/processed" / coffee,
                                 MANIFESTS / f"{coffee}_release.json")
    filter_protected(photos, protect_tanzania(protected))
    deduplicate(photos)

    records = [{"member": p.locator, "publisher_split": p.split, "publisher_label": p.class_name,
                "matching_leafcheck_label": CLASSES[p.class_name] or "", "sha256": p.sha256,
                "decoded_d4_sha256": p.pixel_hash, "disposition": p.reason or "unassigned",
                "possible_match": p.related, "capture_time": p.date, "camera": p.device.rstrip("\0")}
               for p in photos]
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    write_csv(manifest, records)
    totals = defaultdict(Counter)
    for row in records:
        totals[f"{row['publisher_split']}/{row['publisher_label']}"][row["disposition"]] += 1
    report = {"publisher": PUBLISHER, "license": "MIT, dataset release 20 January 2020",
              "archives": provenance, "candidate_count": len(records),
              "frozen_fingerprint": protected[0], "manifest_sha256": file_sha256(manifest),
              "visual_distance": DISTANCE, "by_publisher_split_and_label": totals,
              "capture_time_present": sum(bool(row["capture_time"]) for row in records),
              "roles_assigned": False, "evaluation_run": False}
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    audit(parser.parse_args().output)
