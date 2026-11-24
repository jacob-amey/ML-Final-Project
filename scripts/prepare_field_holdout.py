"""Sample the registered PLDD-UP archives without opening their model scores."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageOps

from prepare_blight_dataset import (
    FROZEN, MANIFESTS, ROOT, RELEASE as BLIGHT, Photo, deduplicate,
    filter_protected, inspect, protected_hashes,
)
from prepare_final_dataset import file_sha256, perceptual_hashes, read_csv, write_csv

SOURCES = MANIFESTS / "leafcheck_field_v1_sources.json"
OUTPUT = ROOT / "data/processed/leafcheck_field_v1"
RAW = ROOT / "data/raw/leafcheck_field_v1"


class RangedZip(io.RawIOBase):
    """A seekable reader of public byte ranges with strict response bounds."""

    def __init__(self, url, size):
        self.url = url
        self.size = size
        self.position = 0
        self.prefetched = (0, b"")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        target = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}[whence] + offset
        if target < 0:
            raise OSError("Negative archive offset")
        self.position = target
        return target

    def fetch(self, start, stop):
        headers = {"Range": f"bytes={start}-{stop - 1}", "User-Agent": "Mozilla/5.0"}
        with urlopen(Request(self.url, headers=headers), timeout=25) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop-1}/{self.size}":
                raise ValueError("Publisher returned a different range or archive size")
            data = response.read(stop - start + 1)
        if len(data) != stop - start:
            raise ValueError("Truncated archive range")
        return data

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.position + size, self.size)
        if end <= self.position:
            return b""
        start = self.position
        self.position = end
        offset, stored = self.prefetched
        if offset <= start and end <= offset + len(stored):
            return stored[start - offset:end - offset]
        return self.fetch(start, end)

    def prefetch(self, start, stop):
        self.prefetched = (start, self.fetch(start, min(stop, self.size)))


def members(record, protocol):
    raw = RangedZip(record["url"], record["size"])
    raw.prefetch(raw.size - 1_000_000, raw.size)
    with ZipFile(raw) as archive:
        info = [x for x in archive.infolist() if not x.is_dir() and
                PurePosixPath(x.filename).suffix.lower() in {".png", ".jpg", ".jpeg"} and
                not any(p.startswith(".") or p.startswith("__MACOSX") for p in PurePosixPath(x.filename).parts)]
        candidates = sorted(info, key=lambda x: hashlib.sha256(
            f"{protocol['code_version']}:{record['name']}:{x.filename}".encode()).hexdigest())
        print(record["name"], len(candidates), "image members", flush=True)
        return candidates[:protocol["candidate_count_per_class"]]


def download(record, sample):
    key = hashlib.sha256(f"{record['name']}:{sample.filename}".encode()).hexdigest()
    path = RAW / "members" / (key + ".bin")
    if path.exists():
        data = path.read_bytes()
    else:
        # Download exactly the local header and compressed member in one range.
        import struct
        import zlib
        raw = RangedZip(record["url"], record["size"])
        start = sample.header_offset
        stop = min(raw.size, start + 30 + sample.compress_size + 4096)
        packed = raw.fetch(start, stop)
        signature, *_, name_length, extra_length = struct.unpack("<4s5H3I2H", packed[:30])
        if signature != b"PK\x03\x04":
            raise ValueError("Unexpected ZIP member header")
        local_length = 30 + name_length + extra_length
        if local_length + sample.compress_size > len(packed):
            packed = raw.fetch(start, start + local_length + sample.compress_size)
        stream = packed[local_length:local_length + sample.compress_size]
        if sample.compress_type == 0:
            data = stream
        elif sample.compress_type == 8:
            data = zlib.decompress(stream, -15)
        else:
            raise ValueError("Unsupported ZIP method in field archive")
        if len(data) != sample.file_size or zlib.crc32(data) != sample.CRC:
            raise ValueError("Unexpected uncompressed member bytes")
        path.write_bytes(data)
    from zlib import crc32
    if len(data) != sample.file_size or crc32(data) != sample.CRC:
        raise ValueError(f"Cached member differs from publisher ZIP metadata: {sample.filename}")
    return data


def include_blight(protected):
    frozen, count, external, raw, exact, paths, phashes, indexes = protected
    visual = phashes.tolist()
    locations = indexes.tolist()
    for row in read_csv(BLIGHT / "manifest.csv"):
        if row["status"] != "included":
            continue
        label = "blight:" + row["image_id"]
        path = BLIGHT / row["output_relative_path"]
        assert file_sha256(path) == row["sha256"]
        raw.setdefault(row["sha256"], label)
        exact.setdefault(row["pixel_d4_sha256"], label)
        with Image.open(path) as image:
            values = perceptual_hashes(ImageOps.exif_transpose(image).convert("RGB"))
        locations.extend([len(paths)] * len(values))
        visual.extend(values)
        paths.append(label)
    return frozen, count, external, raw, exact, paths, np.array(visual, dtype=np.uint64), np.array(locations)


def build(output):
    if output.exists():
        raise FileExistsError("Preserving existing field holdout")
    protocol = json.loads(SOURCES.read_text())
    mapping = json.loads((FROZEN / "class_to_idx.json").read_text())
    (RAW / "members").mkdir(parents=True, exist_ok=True)
    candidates = []
    for entry in protocol["files"]:
        sample = members(entry, protocol)
        with ThreadPoolExecutor(5) as pool:
            chunks = pool.map(lambda item: download(entry, item), sample)
            for i, (item, data) in enumerate(zip(sample, chunks), 1):
                photo = Photo("pldd_up", f"{entry['name']}:{item.filename}", entry["url"], entry["class_name"],
                              entry["name"][:-4], "test", "pldd_up:collection", "", data)
                inspect(photo)
                if photo.date and photo.device:
                    photo.object_id = f"{photo.date}:{photo.device}"
                if not photo.reason and (photo.width * photo.height > 24_000_000 or len(data) > 10 * 1024 * 1024):
                    photo.reason = "above_app_upload_limit"
                candidates.append(photo)
                if i % 24 == 0:
                    print(entry["name"], i, "sampled", flush=True)
    protected = include_blight(protected_hashes())
    filter_protected(candidates, protected)
    deduplicate(candidates)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".field-holdout-") as tmp:
        staging = Path(tmp) / "release"
        staging.mkdir()
        records = []
        for p in candidates:
            if not p.reason:
                dest = Path("test") / p.class_name / (p.sha256 + p.extension)
                (staging / dest).parent.mkdir(parents=True, exist_ok=True)
                (staging / dest).write_bytes(p.content)
                p.output = str(dest)
            records.append({"image_id": "sha256:" + p.sha256, "class_name": p.class_name,
                            "class_id": mapping[p.class_name], "split": "test" if not p.reason else "",
                            "group_id": p.group, "capture_date": p.date, "camera": p.device,
                            "status": "included" if not p.reason else "excluded", "reason": p.reason,
                            "related": p.related, "width": p.width, "height": p.height,
                            "source_locator": p.locator, "source_url": p.url, "sha256": p.sha256,
                            "decoded_d4_sha256": p.pixel_hash, "path": p.output})
        write_csv(staging / "manifest.csv", records)
        stamp = {"manifest_sha256": file_sha256(staging / "manifest.csv"), "sources_sha256": file_sha256(SOURCES),
                 "base_release": protected[0], "master_candidate_records_screened": protected[1],
                 "prior_plantdoc_records_screened": protected[2], "candidate_count": len(records),
                 "included": dict(Counter(p.class_name for p in candidates if not p.reason)),
                 "exclusions": dict(Counter(p.reason for p in candidates if p.reason)),
                 "source_groups": 1, "sealed_for_selection": True}
        (staging / "release.json").write_text(json.dumps(stamp, indent=2) + "\n")
        staging.replace(output)
    print(json.dumps(stamp, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT, type=Path)
    build(parser.parse_args().output)
