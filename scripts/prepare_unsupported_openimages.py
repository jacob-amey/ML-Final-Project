"""Acquire a small, attributable unknown-input challenge for development."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from prepare_final_dataset import read_csv, write_csv


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "data/manifests/leafcheck_unsupported_intake.json"
RAW = ROOT / "data/raw/leafcheck_field_v8/openimages"
REGISTRY = ROOT / "data/manifests/leafcheck_unsupported_openimages_v1_files.csv"
MANIFEST = ROOT / "data/manifests/leafcheck_unsupported_openimages_v1_manifest.csv"


def retrieve(url, limit):
    with urlopen(Request(url, headers={"User-Agent": "LeafCheckResearch/1.0"}), timeout=35) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"File exceeds the size limit: {url}")
    return content


def inventory(plan):
    if REGISTRY.exists():
        raise FileExistsError(REGISTRY)
    for stem, url in plan["metadata"].items():
        path = RAW / f"{stem}.csv"
        if not path.exists():
            path.write_bytes(retrieve(url, 100_000_000))
    include = plan["categories"]
    names = {r["LabelName"]: r["DisplayName"] for r in read_csv(RAW / "classes.csv") if r["DisplayName"] in include}
    positive = defaultdict(list)
    for row in read_csv(RAW / "labels.csv"):
        if row["LabelName"] in names and row["Confidence"] == "1.0":
            positive[row["ImageID"]].append(names[row["LabelName"]])
    options = defaultdict(list)
    for item in read_csv(RAW / "metadata.csv"):
        identifier = item["ImageID"]
        if len(positive.get(identifier, [])) != 1 or not item["Author"] or not item["OriginalLandingURL"]:
            continue
        if not item["License"].startswith(("https://creativecommons.org/licenses/by/", "http://creativecommons.org/licenses/by/")):
            continue
        category = positive[identifier][0]
        options[category].append({
            "image_id": identifier, "category": category, "group": include[category],
            "url": plan["mirror"] + identifier + ".jpg", "source_url": item["OriginalLandingURL"],
            "author": item["Author"], "license": item["License"], "title": item["Title"],
        })
    registry = []
    for name, group in sorted(options.items()):
        order = sorted(group, key=lambda row: hashlib.sha256(row["image_id"].encode()).digest())
        registry.extend(order[:plan["max_photos_per_category"]])
    if set(options) != set(include):
        raise ValueError("Every registered subject needs eligible photos")
    write_csv(REGISTRY, registry)
    print("Catalog support", dict(Counter(item["category"] for item in registry)), flush=True)


def fetch(plan):
    images = RAW / "images"
    images.mkdir(exist_ok=True)
    expected = {r["image_id"]: r["sha256"] for r in read_csv(MANIFEST)} if MANIFEST.exists() else {}

    def download(row):
        path = images / (row["image_id"] + ".jpg")
        if not path.exists():
            content = retrieve(row["url"], plan["max_bytes"])
            if not content.startswith(b"\xff\xd8"):
                raise ValueError(f"Unexpected file format for {row['image_id']}")
            stage = path.with_suffix(".tmp")
            stage.write_bytes(content)
            stage.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if row["image_id"] in expected and expected[row["image_id"]] != digest:
            raise ValueError(f"Pinned mirror bytes changed for {row['image_id']}")
        return digest

    records = read_csv(REGISTRY)
    with ThreadPoolExecutor(4) as pool:
        fingerprints = list(pool.map(download, records))
    stage = RAW / "acquired.csv"
    write_csv(stage, [{**row, "sha256": digest}
                      for row, digest in zip(records, fingerprints, strict=True)])
    print("Downloaded and fingerprinted", len(records), "physical photos", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "fetch"))
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())
    (inventory if args.action == "inventory" else fetch)(plan)
