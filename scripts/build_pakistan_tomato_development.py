"""Screen the camera-dated Pakistan tomato session as one development group."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from inspect_pakistan_tomato import DEFAULT as INTAKE, SOURCE
from prepare_bangladesh_tomato_holdout import DEFAULT as BANGLADESH, protected_with_tom2024
from prepare_tom2024 import MANIFESTS, ROOT, Photo, build_registered_release, extend_protected, inspect, write_csv


DEFAULT = ROOT / "data/processed/leafcheck_pakistan_tomato_dev_v1"


def protection():
    return extend_protected(protected_with_tom2024(), BANGLADESH,
                            MANIFESTS / "leafcheck_bangladesh_tomato_v1_release.json")


def build(intake, output):
    source = json.loads(SOURCE.read_text())
    capture = json.loads((intake / "full_intake.json").read_text())
    rows = capture["sample"]
    if len(rows) != sum(capture["camera_date_folders"].values()) or len({r["member"] for r in rows}) != len(rows):
        raise ValueError("Expected complete and unique camera-tagged archive membership")

    def candidate(row):
        digest = row["sha256"]
        folder = Path(row["member"]).parent.name
        label = source["label_map"][folder]
        content = (intake / (digest + ".jpg")).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest or label != row["declared_label"]:
            raise ValueError(f"Intake image or publisher label changed: {row['member']}")
        photo = Photo("pakistan_tomato", row["member"], source["source"], label, folder, "val",
                      "pakistan_tomato:2025-11-02", "", content)
        inspect(photo)
        filename_stamp = re.search(r"20251102_(\d{6})", Path(row["member"]).name)
        capture_stamp = re.fullmatch(r"2025:11:02 (\d{2}):(\d{2}):(\d{2})", row["capture_time"] or "")
        if not photo.reason and (not filename_stamp or not capture_stamp
                                 or filename_stamp[1] != "".join(capture_stamp.groups())):
            photo.reason = "camera_date_unverified"
        return photo

    with tempfile.TemporaryDirectory() as directory:
        registry = Path(directory) / "files.csv"
        entries = [{**r, "size": "x".join(map(str, r["size"]))} for r in rows]
        write_csv(registry, entries)
        build_registered_release(output, registry, SOURCE, intake, "pakistan_tomato", "val",
                                 protection, prepare_candidate=candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, default=INTAKE)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    build(args.intake, args.output)
