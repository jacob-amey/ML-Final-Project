"""Acquire a fixed preflight of Tanzanian common bean JPEGs with ZIP CRC checks."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image

from inspect_pakistan_tomato import fetch_range
from prepare_blight_dataset import DISTANCE, ROOT, Photo, deduplicate, filter_protected, inspect, protected_hashes


OUTPUT = ROOT / "data/raw/leafcheck_field_v12/tanzania-bean"
PUBLISHER = "https://zenodo.org/records/15685315"
DIRECTORY_BYTES = 2_000_000
SAMPLE_PER_LABEL = 8
REVIEW_EXCLUSIONS = {
    "d448a935cb449b099ed0359eaa36a0ca2348750ebda86aa967a603f891216c63": "suspected_non_bean_leaf",
}
SOURCES = {
    "healthy1.zip": {
        "label": "healthy",
        "archive_size": 1350964354,
        "publisher_md5": "7cf7789c3b8c84ccf89e45234506f91f",
        "directory_sha256": "0035b7afacf77e566d01c8c1aceda941c7d7126dd4cefa0277ded86359cd7c50",
    },
    "rust1.zip": {
        "label": "rust",
        "archive_size": 1134254897,
        "publisher_md5": "3147aecedfcb5a4ff85b899c9882c817",
        "directory_sha256": "7ef2bca75c1bd4d1101b30e19c2f3df4f7604ed33d6695499e0bbac6e8408ed1",
    },
    "anthra.zip": {
        "label": "anthracnose",
        "archive_size": 1760078636,
        "publisher_md5": "b42f3a3fc39acafdc7892eaea9879f2f",
        "directory_sha256": "ad55a58c53890400394aaa96e2b963f2ff0e97f094b42b706e2912e32a9ac40b",
    },
}


def sample_archive(key, metadata, work):
    source = dict(metadata, archive_url=f"{PUBLISHER}/files/{key}?download=1")
    total = source["archive_size"]
    start = total - DIRECTORY_BYTES
    index = OUTPUT / f"{key}.index"
    if not index.exists():
        staged = work / f"{key}.directory"
        fetch_range(source, start, total, staged)
        staged.replace(index)
    directory = index.read_bytes()
    if sha256(directory).hexdigest() != source["directory_sha256"]:
        raise ValueError(f"{key} directory digest mismatch")

    sparse = work / key
    with sparse.open("wb") as stream:
        stream.seek(start)
        stream.write(directory)
    with ZipFile(sparse) as archive:
        members = sorted(archive.infolist(), key=lambda entry: entry.header_offset)
        offsets = [entry.header_offset for entry in members]
        stops = dict(zip(offsets, offsets[1:] + [archive.start_dir]))
        candidates = [entry for entry in members if entry.filename.lower().endswith(".jpg")]
        if len({entry.filename for entry in candidates}) != len(candidates):
            raise ValueError(f"{key} contains repeated JPEG member names")
        selected = sorted(candidates, key=lambda entry: sha256(entry.filename.encode()).digest())[:SAMPLE_PER_LABEL]

    def fetch_member(member):
        offset = member.header_offset
        stop = stops[offset]
        if not 0 < stop - offset < 20_000_000:
            raise ValueError(f"Unexpected member span for {member.filename}")
        span = work / f"{key}-{offset}.member"
        fetch_range(source, offset, stop, span)
        return offset, span.read_bytes()

    with ThreadPoolExecutor(max_workers=4) as pool, sparse.open("r+b") as stream:
        for offset, content in pool.map(fetch_member, selected):
            stream.seek(offset)
            stream.write(content)

    records = []
    with ZipFile(sparse) as archive:
        for member in selected:
            data = archive.read(member)
            digest = sha256(data).hexdigest()
            with Image.open(BytesIO(data)) as photo:
                photo.load()
                if photo.format != "JPEG":
                    raise ValueError(f"Expected JPEG at {member.filename}")
                exif = photo.getexif()
                records.append({"archive": key, "member": member.filename, "label": metadata["label"],
                                "sha256": digest, "dimensions": list(photo.size), "camera": exif.get(272),
                                "capture_time": exif.get_ifd(34665).get(36867) or exif.get(306)})
            (OUTPUT / f"{digest}.jpg").write_bytes(data)
    print(f"{key}: decoded {len(records)} members from {len(candidates)} JPEGs", flush=True)
    return records


def acquire():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".intake.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with TemporaryDirectory(dir=OUTPUT) as temporary:
            work = Path(temporary)
            records = [row for key, source in SOURCES.items() for row in sample_archive(key, source, work)]
            receipt = {"source": PUBLISHER, "license": "CC BY 4.0", "archives": SOURCES,
                       "sampling": "Eight smallest SHA-256 digests of UTF-8 member paths per archive",
                       "role": "quarantined preflight", "evaluation_run": False,
                       "counts": dict(Counter(row["label"] for row in records)), "records": records}
            staged = work / "intake.json"
            staged.write_text(json.dumps(receipt, indent=2) + "\n")
            staged.replace(OUTPUT / "intake.json")


def screen():
    intake = json.loads((OUTPUT / "intake.json").read_text())
    photos = []
    for row in intake["records"]:
        data = (OUTPUT / f"{row['sha256']}.jpg").read_bytes()
        if sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"Original bytes differ for {row['member']}")
        date = row["capture_time"] or "unknown"
        key = row["label"]
        photo = Photo("tanzania-bean", row["member"], PUBLISHER, f"bean__{key}", key,
                      "diagnostic", f"tanzania-bean:{date[:10]}", "", data, date=date, device=row["camera"])
        inspect(photo)
        photo.reason = photo.reason or REVIEW_EXCLUSIONS.get(row["sha256"], "")
        photos.append(photo)
    protected = protected_hashes()
    filter_protected(photos, protected)
    deduplicate(photos)
    records = [dict(row, disposition=photo.reason or "diagnostic", possible_match=photo.related,
                    capture_date_group=photo.group) for row, photo in zip(intake["records"], photos, strict=True)]
    return {"source": PUBLISHER, "role": "development diagnostics only", "license": intake["license"],
            "frozen_fingerprint": protected[0], "master_overlap_candidates_screened": protected[1],
            "plantdoc_overlap_candidates_screened": protected[2], "phash_distance": DISTANCE,
            "sample_count": len(photos), "evaluation_run": False, "records": records}


def verify_uploads(evidence, url):
    import torch

    from leafcheck.inference import Predictor

    torch.set_num_threads(4)
    model = Predictor(ROOT / "artifacts/leafcheck177-v1")
    with urlopen(url + "/api/status", timeout=10) as response:
        runtime = json.load(response)["models"]["final177"]
    assert runtime["checkpoint_sha256"] == model.metadata["checkpoint_sha256"]
    assert runtime["classes"] == model.classes
    bean_labels = [label for label in model.classes if label.startswith("bean__")]
    assert bean_labels == ["bean__halo_blight", "bean__mosaic_virus", "bean__rust"]
    summary = {}
    for row in evidence["records"]:
        if row["disposition"] != "diagnostic":
            continue
        path = OUTPUT / f"{row['sha256']}.jpg"
        request = Request(url + "/api/predict", data=path.read_bytes(), headers={"Content-Type": "image/jpeg"})
        with urlopen(request, timeout=30) as response:
            upload = json.load(response)["models"]["final177"]
        standalone = model.predict(path)
        assert upload["label"] == standalone["label"] and list(upload["class_scores"]) == model.classes
        delta = max(abs(score - upload["class_scores"][label]) for label, score in standalone["class_scores"].items())
        assert delta <= 1e-5
        label = row["label"]
        expected = f"bean__{label}"
        row.update(predicted_label=upload["label"], score=upload["score"], decision=upload["decision"],
                   max_score_difference=delta, supported_condition=expected in bean_labels)
        counts = summary.setdefault(label, Counter(photos=0, suggestions=0, predicted_bean=0,
                                                    correct_supported_label=0))
        counts["photos"] += 1
        counts["suggestions"] += upload["decision"] == "suggestion"
        counts["predicted_bean"] += upload["label"] in bean_labels
        counts["correct_supported_label"] += upload["label"] == expected
    evidence.update(evaluation_run=True, checkpoint_sha256=runtime["checkpoint_sha256"],
                    coverage=bean_labels, diagnostic_counts=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, help="Write a local overlap and annotation screening record")
    parser.add_argument("--verify-url", help="Also replay admitted original bytes against a local app")
    args = parser.parse_args()
    if args.verify_url and args.screen is None:
        parser.error("--verify-url requires --screen")
    if args.screen is None:
        acquire()
        return
    evidence = screen()
    if args.verify_url:
        verify_uploads(evidence, args.verify_url)
    args.screen.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
