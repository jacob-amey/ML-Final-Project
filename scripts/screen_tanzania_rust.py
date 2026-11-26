"""Check the external bean rust recall cohort before either training arm starts."""

from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from inspect_tanzania_bean import OUTPUT
from prepare_blight_dataset import DISTANCE, ROOT, Photo, deduplicate, filter_protected, inspect, protected_hashes
from prepare_final_dataset import file_sha256
from audit_makerere_beans import protect_tanzania


MANIFESTS = ROOT / "data/manifests"
TRAIN = MANIFESTS / "leafcheck_makerere_bean_rust_train.json"
EXPANSION = MANIFESTS / "leafcheck_tanzania_bean_rust_expansion.json"


def screen():
    release = json.loads(TRAIN.read_text())
    cohort = json.loads(EXPANSION.read_text())
    protected = protect_tanzania(protected_hashes())
    fingerprint, _, _, raw, exact, paths, hashes, indexes = protected
    path = ROOT / release["archive"]["path"]
    if fingerprint != release["base_fingerprint"] or file_sha256(path) != release["archive"]["sha256"]:
        raise ValueError("Training protection source changed")
    additions, locations = [], []
    with ZipFile(path) as archive:
        for row in release["members"]:
            locator = "makerere:" + row["member"]
            photo = Photo("makerere", locator, release["source"], "bean__rust", "bean_rust", "train",
                          release["group_id"], "", archive.read(row["member"]))
            inspect(photo)
            if photo.reason or photo.sha256 != row["sha256"]:
                raise ValueError(f"Invalid registered training photo {locator}")
            raw.setdefault(photo.sha256, locator)
            exact.setdefault(photo.pixel_hash, locator)
            additions.extend(photo.perceptual)
            locations.extend([len(paths)] * len(photo.perceptual))
            paths.append(locator)
    protected = (*protected[:6], np.concatenate((hashes, np.asarray(additions, dtype=hashes.dtype))),
                 np.concatenate((indexes, np.asarray(locations, dtype=indexes.dtype))))
    candidates = []
    for row in cohort["records"]:
        source = OUTPUT / (row["sha256"] + ".jpg")
        day = row["capture_time"][:10] if row["capture_time"] else "unknown"
        photo = Photo("tanzania", row["member"], cohort["source"], "bean__rust", row["label"],
                      "val", f"tanzania:{day}", "", source.read_bytes(), date=row["capture_time"] or "")
        inspect(photo)
        if photo.sha256 != row["sha256"] or photo.source_label != "rust":
            raise ValueError(f"Unexpected recall cohort member {row['member']}")
        candidates.append(photo)

    filter_protected(candidates, protected)
    deduplicate(candidates)
    records = [{**row, "disposition": p.reason or "validation", "group": p.group, "possible_match": p.related}
               for row, p in zip(cohort["records"], candidates, strict=True)]
    receipt = {"root_fingerprint": fingerprint, "makerere_training_manifest_sha256": file_sha256(TRAIN),
               "overlap_distance": DISTANCE, "roles": Counter(row["disposition"] for row in records),
               "date_group_counts": Counter(row["group"] for row in records if row["disposition"] == "validation"),
               "source_groups": 1, "training_allowed": False, "records": records}
    target = ROOT / "runs/field_v12/tanzania_rust_validation_screen.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"roles": receipt["roles"], "date_group_counts": receipt["date_group_counts"]}, indent=2))


if __name__ == "__main__":
    screen()
