"""Prepare a prospective coffee field confirmation without running inference."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
from zipfile import ZipFile

from prepare_bangladesh_corn import protection
from prepare_blight_dataset import Photo, inspect
from prepare_final_dataset import file_sha256, write_csv
from prepare_tom2024 import build_registered_release, check_bytes, local_path


ROOT = Path(__file__).resolve().parents[1]
PIN = ROOT / "data/manifests/leafcheck_decafia_v2_source.json"
RAW = ROOT / "data/raw/leafcheck_field_v11/decafia"
OUTPUT = ROOT / "data/processed/leafcheck_decafia_v2"
PREFIX = "CoffeeLeaf-CO-v2"


def build(output):
    plan = json.loads(PIN.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    source = RAW / f"{PREFIX}.zip"
    if not source.exists():
        temporary = source.with_suffix(".download")
        subprocess.run(["curl", "-fsSL", "--max-time", "150", plan["archive_url"],
                        "-o", str(temporary)], check=True)
        if file_sha256(temporary) != plan["archive_sha256"]:
            raise ValueError("Unexpected publisher archive bytes")
        temporary.replace(source)
    if file_sha256(source) != plan["archive_sha256"]:
        raise ValueError("Unexpected publisher archive bytes")
    saved = RAW / "originals"
    saved.mkdir(exist_ok=True)
    pool = defaultdict(list)
    with ZipFile(source) as archive:
        with archive.open(f"{PREFIX}/provenance_manifest.csv") as stream:
            manifest = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig")))
        names = set(archive.namelist())
        archive_images = defaultdict(list)
        for path in names:
            parts = PurePosixPath(path).parts
            if len(parts) == 4 and parts[:2] == (PREFIX, "images") and parts[2] in {"train", "val", "test"}:
                archive_images[parts[-1]].append(path)
        for row in manifest:
            if row["inferred_source"] != "own_field":
                continue
            filename = row["filename"]
            if filename != PurePosixPath(filename).name or row["split"] not in {"train", "val", "test"}:
                raise ValueError("Invalid source path")
            if row["class_ids_present"] == "0":
                label = "coffee__rust"
            elif row["n_annotations"] == "0" and filename.startswith(("SANAS_NUEVAS", "SANAS_SOCORRO")):
                label = "coffee__healthy"
            else:
                continue
            # The v2 archive redistributed images without updating the CSV's split column.
            # Use the unique archive path, then verify bytes against the declared hash.
            matches = archive_images[filename]
            if len(matches) != 1:
                raise ValueError(f"No unique archived image for {filename}")
            member = matches[0]
            archive_split = PurePosixPath(member).parts[2]
            sidecar = f"{PREFIX}/labels/{archive_split}/{PurePosixPath(filename).stem}.txt"
            if sidecar not in names:
                raise ValueError(f"An own-field image or annotation is absent: {filename}")
            pool[label].append((row, member, sidecar))
        if {key: len(items) for key, items in pool.items()} != plan["expected_compatible_catalog_counts"]:
            raise ValueError("Unexpected supported coffee catalog")
        selected = [
            (label, item) for label, group in sorted(pool.items())
            for item in sorted(group, key=lambda entry: sha256(
                ("leafcheck-decafia-v2:" + entry[0]["filename"]).encode()).hexdigest())[:plan["per_label"]]
        ]
        records = []
        for label, (row, member, sidecar) in selected:
            annotation = archive.read(sidecar).decode().splitlines()
            class_ids = {line.split()[0] for line in annotation if line.strip()}
            expected = {"0"} if label == "coffee__rust" else set()
            if class_ids != expected:
                raise ValueError(f"Inconsistent archived label for {member}")
            content = archive.read(member)
            digest = sha256(content).hexdigest()
            if digest != row["sha256"]:
                raise ValueError(f"Inconsistent archived bytes for {member}")
            (saved / f"{digest}.bin").write_bytes(content)
            records.append({"file_id": digest, "member": member, "source_label": "rust boxes" if expected
                            else "healthy background proxy", "class_name": label, "sha256": digest,
                            "size": len(content), "url": plan["archive_url"]})
    registry = RAW / "sample-originals.csv"
    write_csv(registry, records)

    def candidate(row):
        data = local_path(row, saved).read_bytes()
        check_bytes(data, row)
        image = Photo("decafia_v2", row["member"], row["url"], row["class_name"], row["source_label"],
                      "test", "decafia_v2:el_socorro", "", data)
        inspect(image)
        return image

    build_registered_release(output, registry, PIN, saved, "decafia_v2", "test", protection, candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT, type=Path)
    build(parser.parse_args().output)
