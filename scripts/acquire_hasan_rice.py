"""Restore the registered Hasan rice candidates used for source-separated validation."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from inspect_pakistan_tomato import fetch_range
from prepare_blight_dataset import ROOT


RAW = ROOT / "data/raw/leafcheck_field_v13/rice"
PIN = ROOT / "data/manifests/leafcheck_rice_2023_sources.json"
CLASSES = {"Healthy Rice Leaf": "rice__healthy", "Brown Spot": "rice__brown_spot",
           "Sheath Blight": "rice__sheath_blight"}
PUBLISHED_COUNTS = {"Healthy Rice Leaf": 37, "Brown Spot": 55, "Sheath Blight": 58}


def main():
    source = next(r for r in json.loads(PIN.read_text()) if r["name"] == "hasan")
    source["archive_url"] += "?download=true"
    RAW.mkdir(parents=True, exist_ok=True)
    index = RAW / "hasan.index"
    receipt = RAW / "hasan-validation-intake.json"
    with TemporaryDirectory(dir=RAW) as scratch:
        work = Path(scratch)
        sparse = work / "indexed.zip"
        end = source["archive_size"]
        start = end - source["directory_bytes"]
        if not index.exists():
            staged = work / "directory.bin"
            fetch_range(source, start, end, staged)
            staged.replace(index)
        tail = index.read_bytes()
        if sha256(tail).hexdigest() != source["directory_sha256"]:
            raise ValueError("Hasan ZIP directory changed")
        with sparse.open("wb") as stream:
            stream.seek(start)
            stream.write(tail)
        with ZipFile(sparse) as archive:
            members = sorted(archive.infolist(), key=lambda row: row.header_offset)
            offsets = [row.header_offset for row in members]
            stops = dict(zip(offsets, offsets[1:] + [archive.start_dir]))
            selected = [row for row in members if row.filename.lower().endswith(".jpg")
                        and "/Testing data/" in row.filename and Path(row.filename).parent.name in CLASSES]
        counts = Counter(Path(row.filename).parent.name for row in selected)
        if counts != PUBLISHED_COUNTS:
            raise ValueError(f"Unexpected source member counts: {counts}")

        # Already acquired preflight bytes can be reused after a ZIP CRC32 match.
        existing = {}
        preflight = RAW / "hasan-preflight.json"
        if preflight.exists():
            existing = {row["member"]: row["sha256"] for row in json.loads(preflight.read_text())}
        import zlib

        def collect(row):
            digest = existing.get(row.filename)
            if digest:
                cached = (RAW / (digest + ".jpg")).read_bytes()
                if (sha256(cached).hexdigest() == digest and len(cached) == row.file_size
                        and zlib.crc32(cached) == row.CRC):
                    return row, cached, None
            offset, stop = row.header_offset, stops[row.header_offset]
            if not 0 < stop - offset < 20_000_000:
                raise ValueError(f"Unexpected archive span: {row.filename}")
            stage = work / str(offset)
            fetch_range(source, offset, stop, stage)
            return row, None, stage

        cached = {}
        with ThreadPoolExecutor(max_workers=6) as pool, sparse.open("r+b") as stream:
            for count, (member, data, segment) in enumerate(pool.map(collect, selected), 1):
                if data is not None:
                    cached[member.filename] = data
                else:
                    stream.seek(member.header_offset)
                    stream.write(segment.read_bytes())
                if count % 25 == 0:
                    print(f"Acquired {count}/{len(selected)} members", flush=True)
        records = []
        with ZipFile(sparse) as archive:
            for member in selected:
                data = cached.get(member.filename) or archive.read(member)
                digest = sha256(data).hexdigest()
                (RAW / f"{digest}.jpg").write_bytes(data)
                records.append({"member": member.filename, "class_name": CLASSES[Path(member.filename).parent.name],
                                "sha256": digest})
        candidate = work / receipt.name
        candidate.write_text(json.dumps(records, indent=2) + "\n")
        candidate.replace(receipt)
    print(f"Verified {len(records)} publisher-class JPEGs with ZIP CRC checks.", flush=True)


if __name__ == "__main__":
    main()
