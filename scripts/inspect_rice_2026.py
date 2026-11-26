"""Sample pinned rice archives for exact-label and capture-group verification."""

from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from zipfile import ZipFile

from PIL import Image

from prepare_blight_dataset import ROOT, Photo, deduplicate, filter_protected, inspect, protected_hashes
from prepare_final_dataset import file_sha256


PIN = ROOT / "data/manifests/leafcheck_rice_2026_intake.json"
RAW = ROOT / "data/raw/leafcheck_field_v12/rice-2026"
PER_CLASS = 12


def capture_info(stream):
    with Image.open(stream) as image:
        exif = image.getexif()
        captured = str(exif.get_ifd(34665).get(36867) or exif.get(306) or "").rstrip("\0")
        device = str(exif.get(272) or "").rstrip("\0")
        return captured, device


def main():
    pin = json.loads(PIN.read_text())
    metadata, photos = [], []
    inventory = defaultdict(Counter)
    for record in pin["archive_provenance"]:
        archive_path = RAW / record["filename"]
        if archive_path.stat().st_size != record["archive_size"] or file_sha256(archive_path) != record["publisher_archive_sha256"]:
            raise ValueError(f"Unexpected archive bytes: {archive_path}")
        label = record["label"]
        with ZipFile(archive_path) as archive:
            jpeg_members = [name for name in archive.namelist() if name.lower().endswith((".jpg", ".jpeg"))]
            if len(jpeg_members) != record["publisher_count"] or len(set(jpeg_members)) != len(jpeg_members):
                raise ValueError(f"Unexpected original JPEG membership at {archive_path.name}")
            selected = sorted(jpeg_members, key=lambda name: sha256(name.encode()).digest())[:PER_CLASS]
            for member in jpeg_members:
                with archive.open(member) as stream:
                    captured, _ = capture_info(stream)
                inventory[label][captured[:10] or "unknown"] += 1
            for member in selected:
                original = archive.read(member)
                captured, device = capture_info(BytesIO(original))
                locator = f"{record['filename']}::{member}"
                photo = Photo("rice-2026", locator, pin["publisher"], label, label, "candidate",
                              captured[:10] or "unknown", "", original, date=captured, device=device)
                inspect(photo)
                (RAW / (photo.sha256 + ".jpg")).write_bytes(original)
                photo.content = b""
                metadata.append({"archive": record["filename"], "member": member, "label": label,
                                 "capture_time": captured, "camera": device, "sha256": photo.sha256,
                                 "dimensions": [photo.width, photo.height]})
                photos.append(photo)
    protected = protected_hashes()
    filter_protected(photos, protected)
    deduplicate(photos)
    rows = [{**row, "screen": photo.reason or "unassigned", "possible_match": photo.related}
            for row, photo in zip(metadata, photos, strict=True)]
    receipt = {"publisher": pin["publisher"], "role": pin["role"], "license": pin["license"],
               "selection": f"{PER_CLASS} smallest SHA-256 values of archive member paths per label",
               "master_fingerprint": protected[0], "sampled": Counter(row["label"] for row in rows),
               "full_archive_header_capture_days": inventory,
               "capture_groups": Counter(row["capture_time"][:10] or "unknown" for row in rows),
               "dispositions": Counter(row["screen"] for row in rows),
               "records": rows, "training_or_evaluation_run": False}
    staged = RAW / "preflight.tmp.json"
    staged.write_text(json.dumps(receipt, indent=2) + "\n")
    staged.replace(RAW / "rice-2026-preflight.json")
    print(json.dumps({key: receipt[key] for key in ("sampled", "capture_groups", "dispositions")}, indent=2))


if __name__ == "__main__":
    main()
