"""Screen publisher-labeled brassica candidates while preserving raw archives."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from audit_field_preflight import protected_field_sources
from prepare_blight_dataset import DISTANCE, ROOT, Photo, deduplicate, filter_protected, inspect
from prepare_final_dataset import file_sha256, write_csv


PIN = ROOT / "data/manifests/leafcheck_brassica_intake.json"
ORIGINALS = "All-Alt-black_rot-downy_mildew-healthy/ORIGINAL_ALL_COMBINE_CABBAGE_DISEASES.zip"
GHANA_LABELS = {
    "Cabbage_Altenaria_spot": "cabbage__alternaria_leaf_spot",
    "Cabbage - Healthy": "",
    "Cabbage - Black_rot": "cabbage__black_rot",
    "Cabbage downy mildew - Original": "cabbage__downy_mildew",
}
BANGLADESH_LABELS = {
    "Broccoli Alternaria black spot": "broccoli__alternaria_leaf_spot",
    "Cabbage Alternaria leaf spot": "cabbage__alternaria_leaf_spot",
    "Cauliflower Alternaria leaf spot": "cauliflower__alternaria_leaf_spot",
    "Cauliflower Bacterial soft rot": "cauliflower__bacterial_soft_rot",
}


def photos(archive, item, label_map, origin, prefix=""):
    for member in sorted(archive.namelist()):
        path = Path(member)
        if path.suffix.casefold() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        folder = path.parent.name
        class_name = label_map[folder]
        raw = archive.read(member)
        photo = Photo(origin, prefix + member, item["url"], class_name or "unsupported:" + folder, folder,
                      "candidate", origin, "", raw, source_page=item["publisher"], license=item["license"])
        inspect(photo)
        if not photo.reason:
            with Image.open(BytesIO(raw)) as image:
                exif = image.getexif()
                photo.date = exif.get_ifd(34665).get(36867) or photo.date
        photo.content = b""
        yield photo


def audit(output):
    pins = json.loads(PIN.read_text())
    candidates = []
    original_digest = ""
    for source in pins:
        path = ROOT / source["path"]
        if path.stat().st_size != source["bytes"] or file_sha256(path) != source["sha256"]:
            raise ValueError(f"Changed publisher archive: {path}")
        with ZipFile(path) as archive:
            if source["publisher"].endswith("sjmgzhwrxv/1"):
                original = archive.read(ORIGINALS)
                original_digest = sha256(original).hexdigest()
                with ZipFile(BytesIO(original)) as inner:
                    candidates.extend(photos(inner, source, GHANA_LABELS, "ayawli_cabbage", ORIGINALS + "::"))
            else:
                candidates.extend(photos(archive, source, BANGLADESH_LABELS, "ikteshad_brassica"))
    if len(candidates) != 665:
        raise ValueError("Source candidate count changed")
    frozen = protected_field_sources()
    filter_protected(candidates, frozen)
    deduplicate(candidates)
    records = [{"source": p.source, "member": p.locator, "publisher": p.source_page, "source_label": p.source_label,
                "class_hypothesis": p.class_name, "sha256": p.sha256, "decoded_d4_sha256": p.pixel_hash,
                "screen": p.reason or "unassigned", "possible_match": p.related, "capture_time": p.date,
                "camera": p.device.rstrip("\0")} for p in candidates]
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.csv"
    write_csv(manifest, records)
    by_label = defaultdict(Counter)
    exif_groups = defaultdict(Counter)
    for row in records:
        key = f"{row['source']}/{row['class_hypothesis']}"
        by_label[key][row["screen"]] += 1
        if row["capture_time"]:
            exif_groups[key][row["capture_time"][:10]] += 1
    evidence = {"archive_record_sha256": file_sha256(PIN), "nested_originals_sha256": original_digest,
                "manifest_sha256": file_sha256(manifest), "dataset_fingerprint": frozen[0],
                "candidate_count": len(records), "visual_distance": DISTANCE,
                "by_source_and_label": by_label, "capture_day_support": exif_groups,
                "roles_assigned": False, "evaluation_run": False}
    (output / "audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
