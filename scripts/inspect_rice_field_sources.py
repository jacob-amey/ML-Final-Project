"""Restore a fixed, day-limited sample of unmarked rice field photos for intake."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
from zipfile import ZipFile

from PIL import Image

from inspect_pakistan_tomato import fetch_range
from prepare_blight_dataset import DISTANCE, ROOT, Photo, deduplicate, filter_protected, inspect, protected_hashes
from prepare_final_dataset import file_sha256


PIN = ROOT / "data/manifests/leafcheck_rice_sources_v1.json"
RAW = ROOT / "data/raw/leafcheck_field_v12/rice-intake"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ORIGINALS = "RiceLeafDiseaseBD/Original images"


def filename_day(name):
    date = re.search(r"2025[01]\d[0-3]\d", name)
    if date:
        try:
            return datetime.strptime(date.group(), "%Y%m%d").date().isoformat()
        except ValueError:
            return ""
    prefix = Path(name).stem
    if prefix.isdigit() and len(prefix) == 13:
        day = datetime.fromtimestamp(int(prefix) / 1000, timezone.utc).date()
        if day.year == 2025:
            return day.isoformat()
    return ""


def read_metadata(source):
    record = next(row for row in source["related_source_files"] if row["filename"] == "Dataset metadata.xlsx")
    workbook = ROOT / record["path"]
    if file_sha256(workbook) != record["sha256"]:
        raise ValueError("The pinned publisher metadata workbook changed")
    with ZipFile(workbook) as archive:
        strings = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in item.findall(".//s:t", NS))
                  for item in strings.findall("s:si", NS)]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    records = {}
    original_names = Counter()
    for row in sheet.findall("s:sheetData/s:row", NS)[1:]:
        cells = {cell.get("r")[0]: cell.findtext("s:v", namespaces=NS)
                 for cell in row.findall("s:c", NS) if cell.get("t") == "s" and cell.get("r")[0] in "ABC"}
        if set(cells) != set("ABC") or any(value is None for value in cells.values()):
            continue
        category, original, renamed = (shared[int(cells[column])] for column in "ABC")
        key = category.casefold(), renamed.casefold()
        if key in records:
            raise ValueError(f"Repeated renamed image in metadata: {key}")
        records[key] = original
        original_names[original.casefold()] += 1
    if len(records) != 9769:
        raise ValueError("Unexpected RiceLeafDiseaseBD publisher workbook membership")
    return records, original_names


def choose_members(index_path, source, metadata, source_names):
    days = defaultdict(list)
    folders = Counter()
    with ZipFile(index_path) as archive:
        entries = sorted(archive.infolist(), key=lambda row: row.header_offset)
        offsets = [member.header_offset for member in entries]
        stops = dict(zip(offsets, offsets[1:] + [archive.start_dir]))
        for member in entries:
            if not member.filename.startswith(ORIGINALS + "/") or not member.filename.lower().endswith(".jpg"):
                continue
            path = Path(member.filename)
            category = path.parent.name
            folders[category] += 1
            if category not in source["conditions"]:
                continue
            original = metadata.get((category.casefold(), path.name.casefold()))
            if original and source_names[original.casefold()] == 1 and (day := filename_day(original)):
                days[category].append((member, original, day))
    if sum(folders.values()) != len(metadata):
        raise ValueError("ZIP and publisher metadata counts disagree")

    selected = []
    for category in source["conditions"]:
        support = Counter()
        ranked = sorted(days[category], key=lambda item: sha256(item[0].filename.encode()).digest())
        for member, original, day in ranked:
            if support[day] >= source["media_per_capture_day_per_label"]:
                continue
            support[day] += 1
            selected.append((member, original, day, category, stops[member.header_offset]))
            if sum(support.values()) == source["media_per_label"]:
                break
    return selected, folders


def acquire():
    source = json.loads(PIN.read_text())
    metadata, names = read_metadata(source)
    directory = RAW / "riceleafdiseasebd-v3.zip.index"
    stop = source["archive_size"]
    start = stop - source["directory_bytes"]
    with TemporaryDirectory(dir=RAW) as temporary:
        work = Path(temporary)
        index_path = work / "source.zip"
        if not directory.exists():
            staging = work / "index.download"
            fetch_range(source, start, stop, staging)
            staging.replace(directory)
        tail = directory.read_bytes()
        if sha256(tail).hexdigest() != source["directory_sha256"]:
            raise ValueError("The ZIP directory range no longer matches its published version")
        with index_path.open("wb") as stream:
            stream.seek(start)
            stream.write(tail)
        selected, folders = choose_members(index_path, source, metadata, names)

        def member_range(row):
            member, _, _, _, next_offset = row
            offset = member.header_offset
            if not 0 < next_offset - offset < 20_000_000:
                raise ValueError("Unexpected member size")
            stage = work / f"{offset}.data"
            fetch_range(source, offset, next_offset, stage)
            return offset, stage.read_bytes()

        with ThreadPoolExecutor(5) as pool, index_path.open("r+b") as stream:
            for offset, content in pool.map(member_range, selected):
                stream.seek(offset)
                stream.write(content)

        records = []
        with ZipFile(index_path) as archive:
            for member, original, day, category, _ in selected:
                data = archive.read(member)
                digest = sha256(data).hexdigest()
                with Image.open(BytesIO(data)) as image:
                    image.load()
                    if image.format != "JPEG":
                        raise ValueError(f"Unexpected format for {member.filename}")
                    dimensions = list(image.size)
                (RAW / (digest + ".jpg")).write_bytes(data)
                records.append({"archive_member": member.filename, "capture_filename": original,
                                "filename_day": day, "folder_label": category, "sha256": digest,
                                "dimensions": dimensions, "label_hypothesis": source["conditions"][category]})
        receipt = {"publisher": source["publisher"], "original_folder_counts": dict(folders), "records": records,
                   "quota": source["media_per_label"], "quota_per_day": source["media_per_capture_day_per_label"],
                   "date_is_filename_inference": True, "training_or_evaluation_assigned": False}
        staging = work / "preflight.json"
        staging.write_text(json.dumps(receipt, indent=2) + "\n")
        staging.replace(RAW / "riceleafdiseasebd-preflight.json")
        print(f"Downloaded and decoded {len(records)} unmarked intake photos.", flush=True)


def screen(output):
    preflight = json.loads((RAW / "riceleafdiseasebd-preflight.json").read_text())
    photos = []
    for row in preflight["records"]:
        raw = (RAW / (row["sha256"] + ".jpg")).read_bytes()
        photo = Photo("riceleafdiseasebd-v3", row["archive_member"], preflight["publisher"], row["label_hypothesis"],
                      row["folder_label"], "candidate", "filename-day:" + row["filename_day"], "", raw)
        inspect(photo)
        if photo.sha256 != row["sha256"]:
            raise ValueError(f"Original bytes changed for {photo.locator}")
        photos.append(photo)
    protected = protected_hashes()
    filter_protected(photos, protected)
    deduplicate(photos)
    combined = [{**row, "screen": photo.reason or "unassigned", "related": photo.related}
                for row, photo in zip(preflight["records"], photos, strict=True)]
    totals = defaultdict(Counter)
    for item in combined:
        totals[item["folder_label"]][item["screen"]] += 1
    receipt = {"publisher": preflight["publisher"], "master_fingerprint": protected[0], "visual_distance": DISTANCE,
               "catalog_originals": preflight["original_folder_counts"], "by_label": totals,
               "roles_assigned": False, "training_or_evaluation_run": False, "records": combined}
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, help="Check acquired originals against the frozen master and PlantDoc")
    options = parser.parse_args()
    if options.screen:
        screen(options.screen)
    else:
        acquire()
