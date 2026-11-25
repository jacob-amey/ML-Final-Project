"""Build new source-held potato and tomato confirmation data without running inference."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from prepare_ccmt_crop_training import OUTPUT as CCMT, MANIFESTS, ROOT
from prepare_ccmt_crop_training import protection
from prepare_tom2024 import Photo, build_registered_release, check_bytes, extend_protected, fetch_one, inspect
from prepare_tom2024 import local_path
from prepare_final_dataset import file_sha256, read_csv, write_csv


NAME = "leafcheck_crop_confirmation_v1"
INTAKE = MANIFESTS / "leafcheck_crop_confirmation_intake.json"
REGISTRY = MANIFESTS / f"{NAME}_files.csv"
RAW = ROOT / "data/raw/leafcheck_field_v6/crop-confirmation"
RELEASE = ROOT / "data/processed" / NAME


def holeta_archive(source, raw):
    archive = raw / "Local.rar"
    if not archive.exists():
        stage = archive.with_suffix(".download")
        subprocess.run(["curl", "-fLSs", "--retry", "2", "--max-time", "45",
                        source["archive_url"], "-o", str(stage)], check=True)
        if file_sha256(stage) != source["archive_sha256"]:
            raise ValueError("Holeta archive does not match the pinned publisher checksum")
        stage.replace(archive)
    if file_sha256(archive) != source["archive_sha256"]:
        raise ValueError("Holeta archive hash changed")
    return archive


def sample_holeta(source, raw, quota):
    archive = holeta_archive(source, raw)
    members = subprocess.check_output(["bsdtar", "-tf", str(archive)], text=True).splitlines()
    by_label = defaultdict(list)
    for member in members:
        p = Path(member)
        if len(p.parts) == 3 and p.parts[0] == "Local" and p.suffix.casefold() == ".jpg":
            by_label[source["folders"][p.parent.name]].append(member)
    if set(by_label) != set(source["folders"].values()):
        raise ValueError("Holeta folder labels changed")
    result = []
    for label, names in sorted(by_label.items()):
        if len(names) < quota:
            raise ValueError(f"Insufficient Holeta archive members for {label}")
        for member in sorted(names, key=lambda name: sha256(name.encode()).digest())[:quota]:
            content = subprocess.check_output(["bsdtar", "-xOf", str(archive), member])
            digest = sha256(content).hexdigest()
            row = {"source": "holeta", "file_id": digest, "member": member, "class_name": label,
                   "url": source["archive_url"], "size": len(content), "sha256": digest}
            local_path(row, raw).write_bytes(content)
            result.append(row)
    print("Holeta publisher folder counts:", dict((key, len(values)) for key, values in by_label.items()), flush=True)
    return result


def sample_tlid(source, quota):
    crop = source["dataset_id"]
    selected = []
    for folder, label in sorted(source["folders"].items()):
        params = urlencode({"version": 1, "folder_id": folder})
        request = Request(f"https://data.mendeley.com/public-api/datasets/{crop}/files?{params}",
                          headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as response:
            members = json.load(response)
        eligible = [r for r in members if r["content_details"]["content_type"] == "image/jpeg"]
        if len(eligible) < quota:
            raise ValueError(f"Insufficient TLID catalog members for {label}")
        for item in sorted(eligible, key=lambda r: sha256(r["id"].encode()).digest())[:quota]:
            identifier = item["id"]
            info = item["content_details"]
            url = f"https://data.mendeley.com/public-files/datasets/{crop}/files/{identifier}/file_downloaded"
            selected.append({"source": "tlid", "file_id": identifier, "member": f"{folder}/{item['filename']}",
                             "class_name": label, "url": url, "size": info["size"], "sha256": info["sha256_hash"]})
    return selected


def fetch(registry, raw, source):
    rows = read_csv(registry)
    archive = holeta_archive(source, raw)
    for row in rows:
        if row["source"] != "holeta":
            continue
        target = local_path(row, raw)
        data = target.read_bytes() if target.exists() else subprocess.check_output(
            ["bsdtar", "-xOf", str(archive), row["member"]])
        check_bytes(data, row)
        if not target.exists():
            target.write_bytes(data)
    with ThreadPoolExecutor(5) as pool:
        for count, _ in enumerate(pool.map(lambda row: fetch_one(row, raw),
                                           (row for row in rows if row["source"] == "tlid")), 1):
            if count % 32 == 0:
                print(f"Verified {count} TLID files", flush=True)


def build(registry, raw, output, source):
    def photo(row):
        raw_content = local_path(row, raw).read_bytes()
        check_bytes(raw_content, row)
        publisher = row["source"]
        label_folder = Path(row["member"]).parent.name
        source_entry = source["sources"][publisher]
        label = source_entry["folders"][label_folder]
        if label != row["class_name"]:
            raise ValueError("Inconsistent source label")
        p = Photo(publisher, row["member"], source_entry["source"], label, label_folder, "test",
                  f"{publisher}:confirmation_source", "", raw_content)
        inspect(p)
        return p

    def protected():
        from prepare_plantcity_holdout import DEFAULT as PLANTCITY, PAKISTAN
        result = protection()
        for release in (PAKISTAN, PLANTCITY, CCMT):
            result = extend_protected(result, release, MANIFESTS / f"{release.name}_release.json")
        return result

    build_registered_release(output, registry, INTAKE, raw, NAME, "test", protected, prepare_candidate=photo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "fetch", "build"))
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=RELEASE)
    args = parser.parse_args()
    args.raw.mkdir(parents=True, exist_ok=True)
    source = json.loads(INTAKE.read_text())
    with (args.raw / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "inventory":
            if args.registry.exists():
                raise FileExistsError(args.registry)
            quota = source["sample_per_class"]
            rows = sample_holeta(source["sources"]["holeta"], args.raw, quota) + sample_tlid(source["sources"]["tlid"], quota)
            write_csv(args.registry, rows)
        elif args.action == "fetch":
            fetch(args.registry, args.raw, source["sources"]["holeta"])
        else:
            build(args.registry, args.raw, args.output, source)


if __name__ == "__main__":
    main()
