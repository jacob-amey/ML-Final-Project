"""Fetch pinned raw archive members and record source information before assigning roles."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tempfile
from zipfile import ZipFile

from PIL import Image

from prepare_blight_dataset import ROOT


SOURCE = ROOT / "data/manifests/leafcheck_pakistan_tomato_intake.json"
DEFAULT = ROOT / "data/raw/leafcheck_field_v4/pakistan-intake"


def fetch_range(source, start, stop, output):
    expected = f"content-range: bytes {start}-{stop - 1}/{source['archive_size']}"
    headers = output.with_suffix(".headers")
    subprocess.run(
        ["curl", "-fLSs", "--retry", "1", "--max-time", "45", "--max-filesize", str(stop - start),
         "--range", f"{start}-{stop - 1}", "-D", str(headers), "-o", str(output), source["archive_url"]],
        check=True,
    )
    if expected not in headers.read_text().lower() or output.stat().st_size != stop - start:
        raise ValueError(f"Publisher returned the wrong archive bytes for {start}:{stop}")


def inspect(output, all_dated):
    source = json.loads(SOURCE.read_text())
    start = source["archive_size"] - source["directory_bytes"]
    tail = output / "index.bin"
    if not tail.exists():
        stage = output / "index.download"
        fetch_range(source, start, source["archive_size"], stage)
        stage.replace(tail)
    directory = tail.read_bytes()
    if hashlib.sha256(directory).hexdigest() != source["directory_sha256"]:
        raise ValueError("Publisher ZIP directory changed")
    with tempfile.TemporaryDirectory(dir=output) as work:
        scratch = Path(work)
        sparse = scratch / "audit.zip"
        with sparse.open("wb") as stream:
            stream.seek(start)
            stream.write(directory)
        with ZipFile(sparse) as archive:
            all_members = sorted(archive.infolist(), key=lambda m: m.header_offset)
            stops = {a.filename: b.header_offset for a, b in zip(all_members, all_members[1:])}
            raw = [m for m in all_members if "/Dataset (raw)/" in m.filename and not m.is_dir()]
            tagged = [m for m in raw if "20251102_" in Path(m.filename).name]
            picks = []
            for label in source["label_map"]:
                candidates = [m for m in tagged if Path(m.filename).parent.name == label]
                ranked = sorted(candidates, key=lambda m: hashlib.sha256(m.filename.encode()).hexdigest())
                picks.extend(ranked if all_dated else ranked[:source["sample_per_class"]])

        def packed(member):
            span = ranges / str(member.header_offset)
            if not span.exists():
                stage = span.with_suffix(".download")
                fetch_range(source, member.header_offset, stops[member.filename], stage)
                stage.replace(span)
            return member.header_offset, span

        ranges = output / "ranges"
        ranges.mkdir(exist_ok=True)
        with ThreadPoolExecutor(4) as pool:
            with sparse.open("r+b") as stream:
                for count, (offset, span) in enumerate(pool.map(packed, picks), 1):
                    stream.seek(offset)
                    stream.write(span.read_bytes())
                    if count % 20 == 0:
                        print(f"Acquired {count}/{len(picks)} camera-dated archive members", flush=True)

        items = []
        with ZipFile(sparse) as archive:
            for member in picks:
                data = archive.read(member)
                digest = hashlib.sha256(data).hexdigest()
                with Image.open(BytesIO(data)) as image:
                    exif = image.getexif()
                    captured = exif.get_ifd(34665).get(36867)
                    image.load()
                    items.append(dict(member=member.filename, sha256=digest, size=image.size, model=exif.get(272),
                                      capture_time=captured, format=image.format,
                                      declared_label=source["label_map"][Path(member.filename).parent.name]))
                (output / f"{digest}.jpg").write_bytes(data)

    summary = dict(
        all_raw_members=len(raw), raw_folders=Counter(Path(m.filename).parent.name for m in raw),
        camera_date_folders=Counter(Path(m.filename).parent.name for m in tagged), sample=items,
        evaluation_run=False,
    )
    receipt = output / ("full_intake.json" if all_dated else "intake.json")
    stage = receipt.with_suffix(".tmp")
    stage.write_text(json.dumps(summary, indent=2) + "\n")
    stage.replace(receipt)
    print(f"Verified {len(items)} files from {len(tagged)} dated candidates.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--all-dated", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".intake.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inspect(args.output, args.all_dated)
