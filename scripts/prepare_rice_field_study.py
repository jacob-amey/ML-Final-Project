"""Build a versioned exact-label rice study with whole collections kept separate."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZipInfo

from PIL import Image

from audit_field_preflight import protected_field_sources
from prepare_blight_dataset import DISTANCE, ROOT, Photo, deduplicate, filter_protected, inspect
from prepare_final_dataset import file_sha256


RAW = ROOT / "data/raw/leafcheck_field_v13/rice"
TRAIN = ROOT / "data/raw/leafcheck_field_v12/rice-intake"
REGISTRY = ROOT / "data/manifests"
CLASSES = {"rice__healthy", "rice__brown_spot", "rice__sheath_blight"}
TRAIN_QUOTA = 125
BANGLA_QUOTA = 65


def inputs():
    expansion = json.loads((TRAIN / "riceleafdiseasebd-expansion.json").read_text())
    preflight = json.loads((REGISTRY / "leafcheck_rice_field_preflight.json").read_text())
    protected_members = {r["archive_member"] for r in preflight["records"]}
    for row in expansion["records"]:
        if row["archive_member"] in protected_members or row["label_hypothesis"] not in CLASSES:
            raise ValueError("The training expansion contains an ineligible source row")
        path = TRAIN / (row["sha256"] + ".jpg")
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Changed training photo: {path}")
        yield (Photo("munna", row["archive_member"], expansion["publisher"], row["label_hypothesis"],
                     row["folder_label"], "train", "munna:" + row["filename_day"], "", b""),
               path, row["sha256"])

    hasan = json.loads((RAW / "hasan-validation-intake.json").read_text())
    for row in hasan:
        path = RAW / (row["sha256"] + ".jpg")
        with Image.open(path) as image:
            exif = image.getexif()
            day = str(exif.get_ifd(34665).get(36867) or exif.get(306) or "")
            camera = str(exif.get(272) or "").rstrip("\0")
        identity = "hasan:" + day if day else ""
        yield (Photo("hasan", row["member"], "https://data.mendeley.com/datasets/hx6f852hw4/1", row["class_name"],
                     row["class_name"], "validation", "hasan:" + (day[:10] or "unknown"), identity,
                     b"", date=day, device=camera), path, row["sha256"])

    publisher = json.loads((REGISTRY / "leafcheck_banglarice_v2_1_sources.json").read_text())
    for registry in publisher["files"]:
        filename = registry["file"]
        archive = RAW / filename
        if file_sha256(archive) != registry["sha256"]:
            raise ValueError(f"Changed BanglaRice archive {archive}")
        key = archive.stem
        folder, label, quota = {
            "banglarice-healthy": ("Healthy Leaf", "rice__healthy", 500),
            "banglarice-sheath": ("Sheath Blight", "rice__sheath_blight", 418),
        }[key]
        members = sorted((RAW / key / folder).glob("*.jpg"))
        if len(members) != quota:
            raise ValueError("Incomplete extracted BanglaRice archive")
        for path in members:
            file_index = int(path.stem.rsplit("_", 1)[-1])
            # Keep one representative per archive-order bin later. These are proxies, not plant IDs.
            group = f"banglarice:{folder}:bin{(file_index - 1) // 5:03d}"
            yield (Photo("banglarice", str(path.relative_to(RAW / key)), publisher["publisher"], label, folder,
                         "validation", group, "", b""), path, None)


def choose(candidates):
    groups = defaultdict(list)
    for photo, path in candidates:
        groups[photo.source, photo.class_name].append((photo, path))
    included = []
    for (source, _), choices in sorted(groups.items()):
        counts = Counter()
        quota = {"munna": TRAIN_QUOTA, "hasan": 1000, "banglarice": BANGLA_QUOTA}[source]
        kept = 0
        for photo, path in sorted(choices, key=lambda item: item[0].sha256):
            if photo.reason:
                continue
            if kept == quota or (source == "banglarice" and counts[photo.group]):
                photo.reason = "sample_quota"
                continue
            kept += 1
            counts[photo.group] += 1
            included.append((photo, path))
    return included


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(output):
    candidates = []
    for photo, path, expected in inputs():
        photo.content = path.read_bytes()
        inspect(photo)
        photo.content = b""
        if expected and photo.sha256 != expected:
            raise ValueError(f"Changed input photo: {path}")
        candidates.append((photo, path))

    protected = protected_field_sources()
    photos = [photo for photo, _ in candidates]
    filter_protected(photos, protected)
    deduplicate(photos)
    included = choose(candidates)
    train = [(photo, path) for photo, path in included if photo.split == "train"]
    external = [(photo, path) for photo, path in included if photo.split == "validation"]
    support = defaultdict(Counter)
    for photo, _ in included:
        support[photo.source][photo.class_name] += 1
    if (set(support["munna"]) != CLASSES or min(support["munna"].values()) < 90
            or set(support["hasan"]) != CLASSES or min(support["hasan"].values()) < 15
            or set(support["banglarice"]) != {"rice__healthy", "rice__sheath_blight"}
            or min(support["banglarice"].values()) < 35):
        raise ValueError(f"Insufficient cleared support for the registered study: {support}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as scratch:
        staging = Path(scratch) / "release"
        staging.mkdir()
        archive = staging / "training.zip"
        members = []
        with ZipFile(archive, "w") as destination:
            for photo, path in train:
                data = path.read_bytes()
                if sha256(data).hexdigest() != photo.sha256:
                    raise ValueError(f"Input changed during staging: {path}")
                member = f"train/{photo.class_name}/{photo.sha256}.jpg"
                destination.writestr(ZipInfo(member), data)
                members.append({"member": member, "class_name": photo.class_name, "sha256": photo.sha256,
                                "source_locator": photo.locator, "group": photo.group})
        release = {"base_fingerprint": protected[0], "source_split": "train", "role": "experimental training only",
                   "source": "Munna RiceLeafDiseaseBD v3 CC BY 4.0", "class_names": sorted(CLASSES),
                   "archive": {"path": str((output / archive.name).relative_to(ROOT)), "sha256": file_sha256(archive)},
                   "members": members}
        json_write(staging / "train.json", release)
        validation = {"base_fingerprint": protected[0], "role": "source-separated development validation only",
                      "training_sha256": file_sha256(staging / "train.json"), "records": [
            {"source": photo.source, "class_name": photo.class_name, "sha256": photo.sha256,
             "group": photo.group, "member": photo.locator, "path": str(path.relative_to(ROOT))}
            for photo, path in external]}
        json_write(staging / "validation.json", validation)
        rejected = [{"source": photo.source, "label": photo.class_name, "member": photo.locator,
                     "reason": photo.reason, "related": photo.related} for photo in photos if photo.reason]
        audit = {"source_support": support, "candidate_count": len(photos), "excluded": rejected,
                 "visual_distance": DISTANCE, "train_sha256": file_sha256(staging / "train.json"),
                 "validation_sha256": file_sha256(staging / "validation.json")}
        json_write(staging / "audit.json", audit)
        staging.rename(output)
    print(json.dumps(support, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output.resolve())
