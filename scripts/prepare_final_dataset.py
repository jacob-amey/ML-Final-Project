"""Audit and prepare the master plant dataset with reproducible, grouped splits.

Run with the project's Python environment. The master images are read-only inputs.
The final dataset is published after all copied files and split assignments verify.
"""

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import shutil
import time
import urllib.request
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageStat


PROJECT = Path(__file__).resolve().parents[1]
MASTER = PROJECT / "Plant_Dataset_Updated_2026-11-11"
WORK = PROJECT / "data/processed/.leafcheck177_preparation"
OUTPUT = PROJECT / "data/processed/leafcheck177_final"
HF_REVISION = "9e97599868962bd0079b8db4b7f1efa9185fa1e7"
PLANTDOC_REVISION = "5467f6012d78d1c446145d5f582da6096f852ae8"
SEED = 434
MIN_SIDE = 32

# The first eight DCT frequencies are sufficient for a 64-bit perceptual hash.
positions = np.arange(32, dtype=np.float64)
DCT_BASIS = np.cos(np.pi / 32 * (positions + 0.5) * np.arange(8)[:, None])
DCT_BASIS[0] *= 1 / math.sqrt(2)


def read_csv(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_metadata():
    directory = WORK / "metadata"
    directory.mkdir(parents=True, exist_ok=True)
    sources = {
        "leaf-map.json": (
            "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/"
            f"{HF_REVISION}/leaf_grouping/leaf-map.json"
        ),
        "plantdoc-tree.json": (
            "https://api.github.com/repos/pratikkayal/PlantDoc-Dataset/git/trees/"
            f"{PLANTDOC_REVISION}?recursive=1"
        ),
    }
    records = []
    for name, url in sources.items():
        path = directory / name
        if not path.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "LeafCheck-dataset-preparation"})
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        json.loads(path.read_text())
        records.append({"file": name, "url": url, "sha256": file_sha256(path)})
    (directory / "sources.json").write_text(json.dumps(records, indent=2))


def dihedral_views(array):
    """Return the eight exact rotations/reflections as NumPy views."""
    transposed = array.swapaxes(0, 1)
    return [
        array,
        array[:, ::-1],
        array[::-1, :],
        array[::-1, ::-1],
        transposed,
        transposed[:, ::-1],
        transposed[::-1, :],
        transposed[::-1, ::-1],
    ]


def symmetric_positions(length):
    first = {0, length // 8, length // 4, length // 3, length // 2}
    return sorted(first | {length - 1 - value for value in first})


def canonical_pixel_hash(image):
    """Select a canonical orientation using exact, symmetric pixel samples.

    Tied sample signatures are resolved with full pixel hashes. This preserves
    exact rotation/flip invariance without hashing eight full-size images normally.
    """
    array = np.asarray(image)
    height, width = array.shape[:2]
    grid = array[np.ix_(symmetric_positions(height), symmetric_positions(width))]
    views = dihedral_views(array)
    grid_views = dihedral_views(grid)
    signatures = [
        (view.shape[0], view.shape[1], grid_view.tobytes())
        for view, grid_view in zip(views, grid_views)
    ]
    smallest = min(signatures)
    candidates = []
    for view, signature in zip(views, signatures):
        if signature == smallest:
            digest = hashlib.sha256(f"{view.shape[0]}x{view.shape[1]}|".encode())
            digest.update(view.tobytes())
            candidates.append(digest.hexdigest())
    return min(candidates)


def perceptual_hashes(image):
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coefficients = DCT_BASIS @ gray @ DCT_BASIS.T
    signs = (-1.0) ** np.arange(8)
    transformed = [
        coefficients,
        coefficients * signs[None, :],
        coefficients * signs[:, None],
        coefficients * signs[:, None] * signs[None, :],
        coefficients.T,
        coefficients.T * signs[None, :],
        coefficients.T * signs[:, None],
        coefficients.T * signs[:, None] * signs[None, :],
    ]
    values = []
    for transformed_coefficients in transformed:
        flat = transformed_coefficients.ravel()
        bits = np.packbits(flat > np.median(flat[1:]))
        values.append(int.from_bytes(bits.tobytes(), "big"))
    return list(dict.fromkeys(values))


def audit_image(row):
    record = {"master_path": row["output_relative_path"], "errors": []}
    try:
        path = (MASTER / row["output_relative_path"]).resolve()
        if not path.is_relative_to(MASTER.resolve()):
            raise ValueError("Master path escapes the dataset root")
        data = path.read_bytes()
        record["actual_sha256"] = hashlib.sha256(data).hexdigest()
        record["git_blob_sha1"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if record["actual_sha256"] != row["sha256"]:
            record["errors"].append("checksum_mismatch")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as original:
                original.seek(0)
                original.load()
                record.update({
                    "width": original.width,
                    "height": original.height,
                    "format": original.format,
                    "mode": original.mode,
                    "frames": getattr(original, "n_frames", 1),
                    "exif_orientation": original.getexif().get(274, 1),
                })
                image = ImageOps.exif_transpose(original).convert("RGB")
                record["pixel_d4_sha256"] = canonical_pixel_hash(image)
                record["phashes"] = perceptual_hashes(image)
                extrema = image.getextrema()
                record["uniform"] = all(low == high for low, high in extrema)
                record["gray_std"] = ImageStat.Stat(image.convert("L")).stddev[0]
                if min(image.size) < MIN_SIDE:
                    record["errors"].append("minimum_side_below_32")
                if record["uniform"]:
                    record["errors"].append("uniform_image")
        if not row["source_relative_path"] or not row["source_class_name"] or not row["group_id"]:
            record["errors"].append("missing_source_or_group_record")
        if Path(row["output_relative_path"]).parts[:2] != (row["split"], row["class_name"]):
            record["errors"].append("manifest_label_path_mismatch")
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        record["errors"].append("decode_or_file_error")
        record["error_detail"] = f"{type(error).__name__}: {error}"
    return record


def audit(workers):
    WORK.mkdir(parents=True, exist_ok=True)
    fetch_metadata()
    rows = read_csv(MASTER / "manifest.csv")
    snapshot = {
        "manifest_sha256": file_sha256(MASTER / "manifest.csv"),
        "class_mapping_sha256": file_sha256(MASTER / "class_to_idx.json"),
        "minimum_side": MIN_SIDE,
        "audit_version": 1,
    }
    snapshot_path = WORK / "snapshot.json"
    if snapshot_path.exists():
        assert json.loads(snapshot_path.read_text()) == snapshot, "Audit cache belongs to another input or policy"
    else:
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
    cache_path = WORK / "image_audit.jsonl"
    cached = {}
    if cache_path.exists():
        with cache_path.open() as file:
            for line in file:
                record = json.loads(line)
                cached[record["master_path"]] = record
    pending = [row for row in rows if row["output_relative_path"] not in cached]
    print(f"Images already audited {len(cached):,}; remaining {len(pending):,}", flush=True)
    started = time.monotonic()
    with cache_path.open("a") as file, ThreadPoolExecutor(max_workers=workers) as pool:
        for index, record in enumerate(pool.map(audit_image, pending), start=1):
            file.write(json.dumps(record, separators=(",", ":")) + "\n")
            if index % 1000 == 0 or index == len(pending):
                file.flush()
                print(f"Decoded {len(cached) + index:,}/{len(rows):,} images; {time.monotonic() - started:.0f}s this pass", flush=True)
    print("Pixel decoding and checksums completed.", flush=True)


class UnionFind:
    def __init__(self, size):
        self.parents = list(range(size))
        self.sizes = [1] * size

    def find(self, item):
        while self.parents[item] != item:
            self.parents[item] = self.parents[self.parents[item]]
            item = self.parents[item]
        return item

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first == second:
            return
        if self.sizes[first] < self.sizes[second]:
            first, second = second, first
        self.parents[second] = first
        self.sizes[first] += self.sizes[second]


def plantvillage_class(official):
    aliases = {
        "Apple_Frogeye Spot": "apple__black_rot",
        "Apple___Apple_scab": "apple__scab",
        "Apple___Cedar_apple_rust": "apple__rust",
        "Grape___Esca_(Black_Measles)": "grape__black_measles",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape__leaf_blight_isariopsis_leaf_spot",
        "Orange___Haunglongbing_(Citrus_greening)": "citrus__greening_disease",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "tomato__yellow_leaf_curl_virus",
        "Tomato___Tomato_mosaic_virus": "tomato__mosaic_virus",
        "Tomato___Spider_mites Two-spotted_spider_mite": "tomato__spider_mites_two_spotted_spider_mite",
    }
    if official in aliases:
        return aliases[official]
    crop, condition = official.split("___", 1)
    crop = {"Cherry_(including_sour)": "cherry", "Pepper,_bell": "bell_pepper"}.get(crop, crop)
    return crop.lower() + "__" + re.sub(r"[^a-z0-9]+", "_", condition.lower()).strip("_")


def specimen_keys(row, leaf_map):
    filename = Path(row["source_relative_path"]).name
    identifier = filename.replace("_final_masked", "").split("___")[-1]
    identifier = identifier.split("copy")[0]
    identifier = re.sub(r"\.(?:jpg|jpeg|png)$", "", identifier, flags=re.IGNORECASE).strip()
    documented = [
        value for value in leaf_map.get(identifier.lower(), [])
        if plantvillage_class(value.split(":::")[0]) == row["class_name"]
    ]
    keys = ["plantvillage:" + value for value in documented]
    # A named Leaf number is a conservative family hint within its capture prefix.
    # Decimal views and different named days remain together. It is not a verified
    # specimen ID and is labelled separately in the final manifest.
    family = re.match(r"(.*?)\bleaf\s*(\d+)(?:\D|$)", identifier, flags=re.IGNORECASE)
    if "___" in filename and family:
        prefix = re.sub(r"[^a-z0-9]+", "_", family.group(1).lower()).strip("_")
        keys.append(f"filename_leaf_family:{row['class_name']}:{prefix}:{family.group(2)}")
    return keys, documented


def previous_exposures():
    snapshot_path = WORK / "previous_exposures.json"
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        return set(snapshot["sha256"]), snapshot["record_sources"]
    hashes = set()
    sources = []
    for path in sorted((PROJECT / "runs/leafcheck_small").glob("*/selected_images.csv")):
        hashes.update(row["sha256"] for row in read_csv(path))
        sources.append(str(path.relative_to(PROJECT)))
    starter_manifest = PROJECT / "Plant_CNN_Starter_300/manifest.csv"
    if starter_manifest.exists():
        hashes.update(row["sha256"] for row in read_csv(starter_manifest))
        sources.append(str(starter_manifest.relative_to(PROJECT)))
    # Discover recorded dataset manifests from completed experiment runs as well.
    for path in sorted((PROJECT / "runs").glob("*/20*/dataset_manifest.csv")):
        hashes.update(row["sha256"] for row in read_csv(path))
        sources.append(str(path.relative_to(PROJECT)))
    snapshot_path.write_text(json.dumps({"sha256": sorted(hashes), "record_sources": sources}, indent=2))
    return hashes, sources


def previous_inventory():
    """Freeze earlier human review/grouping evidence, without its old ID policy."""
    snapshot_path = WORK / "prior_inventory.csv"
    source = PROJECT / "data/manifests/image_inventory.csv"
    if not snapshot_path.exists() and source.exists():
        shutil.copyfile(source, snapshot_path)
    return read_csv(snapshot_path) if snapshot_path.exists() else []


def load_audited_rows():
    records = {}
    with (WORK / "image_audit.jsonl").open() as file:
        for line in file:
            record = json.loads(line)
            records[record["master_path"]] = record
    rows = read_csv(MASTER / "manifest.csv")
    assert {row["output_relative_path"] for row in rows} == set(records), "Run the complete audit first"
    for row in rows:
        row["audit"] = records[row["output_relative_path"]]
    return rows


def initial_groups(rows):
    leaf_map = json.loads((WORK / "metadata/leaf-map.json").read_text())
    plantdoc_tree = json.loads((WORK / "metadata/plantdoc-tree.json").read_text())
    plantdoc_blobs = defaultdict(list)
    for entry in plantdoc_tree["tree"]:
        if entry["type"] == "blob" and entry["path"].startswith(("train/", "test/")):
            plantdoc_blobs[entry["sha"]].append(entry["path"])
    exposed, exposure_sources = previous_exposures()
    prior_by_hash = defaultdict(list)
    for record in previous_inventory():
        prior_by_hash[record["sha256"]].append(record)
    rejected_pixels = {}
    for row in rows:
        for record in prior_by_hash.get(row["sha256"], []):
            if record["status"] == "excluded" and record["notes"].startswith((
                "Multi-panel", "Visible disease name", "Source filename explicitly",
                "Illustration", "Same photograph also labeled",
            )):
                rejected_pixels[row["audit"].get("pixel_d4_sha256")] = record["notes"]
    groups = UnionFind(len(rows))
    seen = {}
    for index, row in enumerate(rows):
        keys, documented = specimen_keys(row, leaf_map)
        row["documented_leaf_ids"] = documented
        row["filename_family_ids"] = [key for key in keys if key.startswith("filename_leaf_family:")]
        row["plantdoc_source_paths"] = plantdoc_blobs.get(row["audit"].get("git_blob_sha1"), [])
        row["previously_used"] = row["sha256"] in exposed
        row["offline_augmentation"] = "flip" in row["source_relative_path"].lower()
        row["prior_review_exclusion"] = rejected_pixels.get(row["audit"].get("pixel_d4_sha256"), "")
        row["prior_review_group_ids"] = []
        for record in prior_by_hash.get(row["sha256"], []):
            if record.get("group_id"):
                key = "prior_inventory:" + record["source"] + ":" + record["source_version"] + ":" + record["group_id"]
                keys.append(key)
                row["prior_review_group_ids"].append(key)
        keys.append("master_group:" + row["group_id"])
        keys.append("sha256:" + row["sha256"])
        pixel_hash = row["audit"].get("pixel_d4_sha256")
        if pixel_hash:
            keys.append("pixel_d4:" + pixel_hash)
        for key in keys:
            if key in seen:
                groups.union(index, seen[key])
            else:
                seen[key] = index
    print("Recovered documented leaf IDs for", sum(bool(r["documented_leaf_ids"]) for r in rows), "images", flush=True)
    print("Recovered filename leaf families for", sum(bool(r["filename_family_ids"]) for r in rows), "images", flush=True)
    return groups, exposure_sources


def find_near_pairs(rows, groups):
    """Find rotation/flip-aware pHash candidates within three Hamming bits.

    Four disjoint 16-bit chunks guarantee an exact chunk match for any such pair.
    Perceptual similarity alone does not trigger exclusion or relabelling.
    """
    path = WORK / "perceptual_candidates.csv"
    if path.exists():
        return read_csv(path)
    buckets = defaultdict(list)
    pairs = []
    for index, row in enumerate(rows):
        if row["audit"]["errors"]:
            continue
        hashes = row["audit"]["phashes"]
        matches = set()
        for value in hashes:
            for part in range(4):
                matches.update(buckets.get((part, (value >> (part * 16)) & 65535), []))
        for other in sorted(matches):
            if groups.find(index) == groups.find(other):
                continue
            other_hash = rows[other]["audit"]["phashes"][0]
            distance = min((value ^ other_hash).bit_count() for value in hashes)
            if distance <= 3:
                pairs.append({
                    "first": row["output_relative_path"],
                    "second": rows[other]["output_relative_path"],
                    "first_class": row["class_name"],
                    "second_class": rows[other]["class_name"],
                    "distance": distance,
                })
        value = hashes[0]
        for part in range(4):
            buckets[(part, (value >> (part * 16)) & 65535)].append(index)
        if (index + 1) % 10000 == 0:
            print(f"Perceptual screen {index + 1:,}/{len(rows):,}; candidates {len(pairs):,}", flush=True)
    write_csv(path, pairs, fields=["first", "second", "first_class", "second_class", "distance"])
    return pairs


@lru_cache(maxsize=512)
def thumbnail(master_path):
    with Image.open(MASTER / master_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32)


def score_pair(pair):
    first = thumbnail(pair["first"])
    second = thumbnail(pair["second"])
    errors = []
    for view in dihedral_views(second):
        difference = first - view
        errors.append((float(np.abs(difference).mean()), float(np.sqrt(np.square(difference).mean()))))
    mae, rmse = min(errors)
    return {**pair, "mean_absolute_error": round(mae, 6), "root_mean_square_error": round(rmse, 6)}


def near_scores(rows, groups, pairs, workers):
    path = WORK / "perceptual_similarity.csv"
    if path.exists():
        return read_csv(path)
    lookup = {row["output_relative_path"]: index for index, row in enumerate(rows)}
    labels = defaultdict(set)
    for index, row in enumerate(rows):
        labels[groups.find(index)].add(row["class_name"])
    candidates = [
        pair for pair in pairs
        if len(labels[groups.find(lookup[pair["first"]])]) == 1
        and len(labels[groups.find(lookup[pair["second"]])]) == 1
    ]
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(score_pair, candidates), start=1):
            results.append(result)
            if index % 1000 == 0 or index == len(candidates):
                print(f"Compared pixels for {index:,}/{len(candidates):,} perceptual candidates", flush=True)
    fields = ["first", "second", "first_class", "second_class", "distance", "mean_absolute_error", "root_mean_square_error"]
    write_csv(path, results, fields=fields)
    return results


def apply_near_groups(rows, groups, scores):
    lookup = {row["output_relative_path"]: index for index, row in enumerate(rows)}
    review_path = WORK / "manual_review.csv"
    reviewed = {}
    if review_path.exists():
        reviewed = {(row["first"], row["second"]): row for row in read_csv(review_path)}
    pending = []
    links = []
    for score in scores:
        decision = reviewed.get((score["first"], score["second"]))
        if decision and decision["decision"] == "same_source":
            groups.union(lookup[score["first"]], lookup[score["second"]])
            links.append({**score, "decision": "visually_reviewed_same_source"})
            continue
        if decision and decision["decision"] == "different_images":
            continue
        if decision and decision["decision"] == "quarantine_uncertain":
            # Do not infer a biological diagnosis or join two uncertain specimens.
            # Their existing related groups are excluded in build_records instead.
            continue
        if float(score["mean_absolute_error"]) > 1 or float(score["root_mean_square_error"]) > 3:
            continue
        first, second = lookup[score["first"]], lookup[score["second"]]
        if rows[first]["class_name"] == rows[second]["class_name"]:
            groups.union(first, second)
            links.append({**score, "decision": "group_same_label_near_copy"})
        else:
            pending.append(score)
    write_csv(WORK / "pending_review.csv", pending, fields=list(scores[0]) if scores else ["first", "second"])
    write_csv(WORK / "near_group_links.csv", links, fields=(list(scores[0]) + ["decision"]) if scores else ["first", "second", "decision"])
    return pending, links


def group(workers):
    rows = load_audited_rows()
    groups, exposure_sources = initial_groups(rows)
    pairs = find_near_pairs(rows, groups)
    scores = near_scores(rows, groups, pairs, workers)
    pending, links = apply_near_groups(rows, groups, scores)
    components = defaultdict(list)
    for index, row in enumerate(rows):
        components[groups.find(index)].append(index)
    conflicts = [indices for indices in components.values() if len({rows[i]["class_name"] for i in indices}) > 1]
    summary = {
        "audited_images": len(rows),
        "audit_errors": dict(Counter(reason for row in rows for reason in row["audit"]["errors"])),
        "related_groups": len(components),
        "conflict_groups": len(conflicts),
        "images_in_conflict_groups": sum(map(len, conflicts)),
        "documented_leaf_images": sum(bool(row["documented_leaf_ids"]) for row in rows),
        "filename_leaf_family_images": sum(bool(row["filename_family_ids"]) for row in rows),
        "plantdoc_traceable_images": sum(bool(row["plantdoc_source_paths"]) for row in rows),
        "perceptual_candidates": len(pairs),
        "cross_label_perceptual_candidates": sum(pair["first_class"] != pair["second_class"] for pair in pairs),
        "near_copy_group_links": len(links),
        "pending_cross_label_reviews": len(pending),
        "exposure_record_sources": exposure_sources,
    }
    (WORK / "grouping_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def select_holdout(candidates, target_images, leave_groups, group_records):
    selected = []
    count = 0
    for key in sorted(candidates, key=lambda key: -len(group_records[key]["included_indices"])):
        if len(candidates) - len(selected) <= leave_groups:
            break
        size = len(group_records[key]["included_indices"])
        if count + size <= target_images:
            selected.append(key)
            count += size
    if not selected and len(candidates) > leave_groups:
        selected = [min(candidates, key=lambda key: len(group_records[key]["included_indices"]))]
    return selected


def assign_splits(group_records):
    """Allocate groups per class with 70/20/10 image targets and fresh holdouts."""
    by_class = defaultdict(list)
    for key, record in group_records.items():
        if record["included_indices"]:
            by_class[record["class_name"]].append(key)
    rng = random.Random(SEED)
    assignments = {}
    for name in sorted(by_class):
        keys = sorted(by_class[name])
        rng.shuffle(keys)
        forced = [key for key in keys if group_records[key]["training_only"]]
        fresh = [key for key in keys if not group_records[key]["training_only"]]
        if not forced:
            reserved = max(fresh, key=lambda key: len(group_records[key]["included_indices"]))
            forced.append(reserved)
            fresh.remove(reserved)
        for key in forced:
            assignments[key] = "train"
        images = sum(len(group_records[key]["included_indices"]) for key in keys)
        if len(fresh) >= 2:
            testing = select_holdout(fresh, max(1, round(images * 0.1)), 1, group_records)
            for key in testing:
                assignments[key] = "test"
                fresh.remove(key)
            validating = select_holdout(fresh, max(1, round(images * 0.2)), 0, group_records)
            for key in validating:
                assignments[key] = "val"
                fresh.remove(key)
        elif fresh:
            assignments[fresh.pop()] = "test"
        for key in fresh:
            assignments[key] = "train"
    return assignments


def build_records(rows, groups, links):
    components = defaultdict(list)
    for index in range(len(rows)):
        components[groups.find(index)].append(index)
    original_mapping = json.loads((MASTER / "class_to_idx.json").read_text())
    source_mapping = {row["source_folder"]: row["standardized_class_name"] for row in read_csv(MASTER / "label_mapping.csv")}
    reasons = {}
    duplicate_of = {}
    group_records = {}
    path_to_index = {row["output_relative_path"]: index for index, row in enumerate(rows)}
    near_roots = {groups.find(path_to_index[link["first"]]) for link in links}
    quarantined_roots = set()
    review_path = WORK / "manual_review.csv"
    if review_path.exists():
        for review in read_csv(review_path):
            if review["decision"] == "quarantine_uncertain":
                for key in ("first", "second"):
                    quarantined_roots.add(groups.find(path_to_index[review[key]]))

    for root, indices in components.items():
        labels = {rows[index]["class_name"] for index in indices}
        stable_id = "grp_" + hashlib.sha256(
            "\n".join(sorted(rows[index]["sha256"] for index in indices)).encode()
        ).hexdigest()[:24]
        methods = set()
        if any(rows[index]["documented_leaf_ids"] for index in indices):
            methods.add("documented_leaf")
        if any(rows[index]["filename_family_ids"] for index in indices):
            methods.add("filename_leaf_family")
        if any(rows[index]["prior_review_group_ids"] for index in indices):
            methods.add("prior_review_inventory")
        if len({rows[index]["group_id"] for index in indices}) < len(indices):
            methods.add("source_duplicate_or_augmentation")
        if root in near_roots:
            methods.add("reviewed_or_strict_near_copy")
        training_reasons = []
        if any(rows[index]["previously_used"] for index in indices):
            training_reasons.append("previous_experiment_exposure")
        if any(rows[index]["offline_augmentation"] for index in indices):
            training_reasons.append("offline_augmentation_family")
        candidates = []
        for index in indices:
            row = rows[index]
            row_reasons = list(row["audit"]["errors"])
            if row["prior_review_exclusion"]:
                row_reasons.append("prior_visual_review_content_or_annotation_issue")
            source_path = Path(row["source_relative_path"])
            valid_label = (
                row["class_name"] in original_mapping
                and str(original_mapping[row["class_name"]]) == row["class_id"]
                and source_mapping.get(row["source_class_name"]) == row["class_name"]
                and not source_path.is_absolute()
                and ".." not in source_path.parts
                and source_path.parts
                and source_path.parts[0] == row["source_class_name"]
            )
            if not valid_label:
                row_reasons.append("invalid_or_untraceable_label_record")
            if len(labels) > 1:
                row_reasons.append("conflicting_labels_in_related_group")
            if root in quarantined_roots:
                row_reasons.append("unresolved_cross_label_visual_review")
            if row_reasons:
                reasons[index] = sorted(set(row_reasons))
            else:
                candidates.append(index)

        pixel_groups = defaultdict(list)
        for index in candidates:
            pixel_groups[rows[index]["audit"]["pixel_d4_sha256"]].append(index)
        included = []
        for identical in pixel_groups.values():
            selected = min(identical, key=lambda index: (
                rows[index]["offline_augmentation"],
                not bool(rows[index]["documented_leaf_ids"]),
                -(rows[index]["audit"]["width"] * rows[index]["audit"]["height"]),
                rows[index]["source_relative_path"],
            ))
            included.append(selected)
            if len(identical) > 1:
                methods.add("exact_decoded_rotation_flip_identity")
            for index in identical:
                if index != selected:
                    reason = "duplicate_file" if rows[index]["sha256"] == rows[selected]["sha256"] else "duplicate_decoded_rotation_or_flip"
                    reasons[index] = [reason]
                    duplicate_of[index] = rows[selected]["output_relative_path"]
        group_records[stable_id] = {
            "source_indices": indices,
            "included_indices": included,
            "class_name": rows[included[0]]["class_name"] if included else "",
            "grouping_methods": "|".join(sorted(methods)) or "traceable_content_id",
            "training_only": bool(training_reasons),
            "training_only_reason": "|".join(training_reasons),
        }

    assignments = assign_splits(group_records)
    retained_classes = sorted({record["class_name"] for record in group_records.values() if record["included_indices"]})
    mapping = {name: index for index, name in enumerate(retained_classes)}
    records = [None] * len(rows)
    for group_id, group_record in group_records.items():
        included = set(group_record["included_indices"])
        for index in group_record["source_indices"]:
            row = rows[index]
            is_included = index in included
            split = assignments[group_id] if is_included else ""
            final_path = (Path(split) / row["class_name"] / Path(row["output_relative_path"]).name).as_posix() if is_included else ""
            records[index] = {
                "image_id": "sha256:" + row["audit"].get("actual_sha256", row["sha256"]),
                "class_id": mapping[row["class_name"]] if is_included else "",
                "class_name": row["class_name"],
                "split": split,
                "output_relative_path": final_path,
                "status": "included" if is_included else "excluded",
                "exclusion_reasons": "|".join(reasons.get(index, [])),
                "duplicate_of_master_path": duplicate_of.get(index, ""),
                "group_id": group_id,
                "grouping_methods": group_record["grouping_methods"],
                "training_only": group_record["training_only"],
                "training_only_reason": group_record["training_only_reason"],
                "documented_leaf_ids": "|".join(row["documented_leaf_ids"]),
                "filename_leaf_family_ids": "|".join(row["filename_family_ids"]),
                "prior_review_group_ids": "|".join(row["prior_review_group_ids"]),
                "prior_review_exclusion_note": row["prior_review_exclusion"],
                "previously_used": row["previously_used"],
                "offline_augmentation": row["offline_augmentation"],
                "master_relative_path": row["output_relative_path"],
                "master_split": row["split"],
                "master_class_id": row["class_id"],
                "source_class_name": row["source_class_name"],
                "source_relative_path": row["source_relative_path"],
                "plantdoc_source_paths": "|".join(row["plantdoc_source_paths"]),
                "sha256": row["audit"].get("actual_sha256", ""),
                "pixel_d4_sha256": row["audit"].get("pixel_d4_sha256", ""),
                "width": row["audit"].get("width", ""),
                "height": row["audit"].get("height", ""),
                "format": row["audit"].get("format", ""),
                "mode": row["audit"].get("mode", ""),
                "frames": row["audit"].get("frames", ""),
            }
    assert len(records) == len(rows)
    return records, group_records, mapping


def verify_records(records, mapping):
    included = [record for record in records if record["status"] == "included"]
    assert all(record["exclusion_reasons"] for record in records if record["status"] == "excluded")
    paths, hashes, pixels = set(), set(), set()
    group_splits = defaultdict(set)
    evidence_splits = defaultdict(set)
    counts = Counter()
    for row in included:
        assert row["split"] in {"train", "val", "test"}
        assert int(row["class_id"]) == mapping[row["class_name"]]
        assert row["image_id"] == "sha256:" + row["sha256"]
        assert Path(row["output_relative_path"]).parts[:2] == (row["split"], row["class_name"])
        assert row["output_relative_path"] not in paths
        assert row["sha256"] not in hashes
        assert row["pixel_d4_sha256"] not in pixels
        paths.add(row["output_relative_path"])
        hashes.add(row["sha256"])
        pixels.add(row["pixel_d4_sha256"])
        group_splits[row["group_id"]].add(row["split"])
        for field in ("documented_leaf_ids", "filename_leaf_family_ids", "prior_review_group_ids"):
            for identifier in row[field].split("|"):
                if identifier:
                    evidence_splits[(field, identifier)].add(row["split"])
        counts[row["split"], row["class_name"]] += 1
        if str(row["training_only"]) == "True":
            assert row["split"] == "train"
        if row["split"] != "train":
            assert str(row["previously_used"]) == "False"
            assert str(row["offline_augmentation"]) == "False"
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert all(len(splits) == 1 for splits in evidence_splits.values())
    assert set(mapping) == {row["class_name"] for row in included}
    assert all(counts["train", name] > 0 for name in mapping)
    assert sorted(mapping.values()) == list(range(len(mapping)))
    return included


def unreviewed_cross_label_components(rows, groups, records, scores, manual_reviews):
    """Report unresolved component pairs, avoiding repeated views of reviewed pairs."""
    lookup = {row["output_relative_path"]: index for index, row in enumerate(rows)}

    def roots(pair):
        return tuple(sorted(groups.find(lookup[pair[key]]) for key in ("first", "second")))

    reviewed_components = {roots(review) for review in manual_reviews}
    retained_roots = {groups.find(index) for index, row in enumerate(records) if row["status"] == "included"}
    unreviewed = {}
    for score in scores:
        if score["first_class"] == score["second_class"]:
            continue
        pair_roots = roots(score)
        if pair_roots in reviewed_components or pair_roots[0] == pair_roots[1] or not set(pair_roots).issubset(retained_roots):
            continue
        if pair_roots not in unreviewed or float(score["mean_absolute_error"]) < float(unreviewed[pair_roots]["mean_absolute_error"]):
            unreviewed[pair_roots] = score
    return list(unreviewed.values())


def prepare(workers):
    if OUTPUT.exists():
        print("Final dataset already exists. Verifying its contents.", flush=True)
        verify(workers)
        return
    rows = load_audited_rows()
    groups, exposure_sources = initial_groups(rows)
    pairs = find_near_pairs(rows, groups)
    scores = near_scores(rows, groups, pairs, workers)
    pending, links = apply_near_groups(rows, groups, scores)
    assert not pending, f"Review {len(pending)} cross-label candidates in {WORK / 'pending_review.csv'} first"
    records, group_records, mapping = build_records(rows, groups, links)
    included = verify_records(records, mapping)
    counts = Counter(row["split"] for row in included)
    reason_counts = Counter(reason for row in records for reason in row["exclusion_reasons"].split("|") if reason)
    print("Retained images", len(included), "classes", len(mapping), "splits", dict(counts), flush=True)
    print("Exclusion reasons", dict(reason_counts), flush=True)

    staging = OUTPUT.with_name(OUTPUT.name + ".staging")
    staging.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((WORK / "snapshot.json").read_text())
    snapshot["preparation_policy"] = {"seed": SEED, "ratios": [0.7, 0.2, 0.1],
                                      "minimum_side": MIN_SIDE, "policy_version": 2}
    snapshot["evidence_sha256"] = {
        name: file_sha256(WORK / name)
        for name in ("manual_review.csv", "prior_inventory.csv", "previous_exposures.json")
        if (WORK / name).exists()
    }
    snapshot_path = staging / "preparation_snapshot.json"
    if snapshot_path.exists():
        assert json.loads(snapshot_path.read_text()) == snapshot
    else:
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
    write_csv(staging / "manifest.csv", records)
    write_csv(staging / "exclusions.csv", [row for row in records if row["status"] == "excluded"], fields=list(records[0]))
    (staging / "class_to_idx.json").write_text(json.dumps(mapping, indent=2))
    for split in ("train", "val", "test"):
        for name in mapping:
            (staging / split / name).mkdir(parents=True, exist_ok=True)

    def copy_and_verify(row):
        destination = staging / row["output_relative_path"]
        if not destination.exists() or file_sha256(destination) != row["sha256"]:
            temporary = destination.with_suffix(destination.suffix + ".copying")
            shutil.copyfile(MASTER / row["master_relative_path"], temporary)
            assert file_sha256(temporary) == row["sha256"], temporary
            temporary.replace(destination)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(copy_and_verify, included), start=1):
            if index % 5000 == 0 or index == len(included):
                print(f"Copied and verified {index:,}/{len(included):,} retained images", flush=True)

    summaries = []
    for name, class_id in mapping.items():
        class_records = [row for row in included if row["class_name"] == name]
        summary = {"class_id": class_id, "class_name": name, "total": len(class_records)}
        for split in ("train", "val", "test"):
            subset = [row for row in class_records if row["split"] == split]
            summary[split] = len(subset)
            summary[split + "_groups"] = len({row["group_id"] for row in subset})
        summary["documented_leaf_images"] = sum(bool(row["documented_leaf_ids"]) for row in class_records)
        summaries.append(summary)
    write_csv(staging / "split_summary.csv", summaries)
    metadata = staging / "metadata"
    metadata.mkdir(exist_ok=True)
    shutil.copytree(WORK / "metadata", metadata / "upstream", dirs_exist_ok=True)
    for name in ("snapshot.json", "previous_exposures.json", "prior_inventory.csv", "image_audit.jsonl", "perceptual_candidates.csv", "perceptual_similarity.csv", "manual_review.csv", "near_group_links.csv", "additional_cross_label_review.csv", "additional_priority_review.csv", "other_crop_review.csv", "grouping_summary.json"):
        if (WORK / name).exists():
            shutil.copyfile(WORK / name, metadata / name)
    for name in ("label_mapping.csv", "dataset_info.json", "README.md"):
        shutil.copyfile(MASTER / name, metadata / ("master_" + name))
    manual_reviews = read_csv(WORK / "manual_review.csv") if (WORK / "manual_review.csv").exists() else []
    unreviewed_components = unreviewed_cross_label_components(rows, groups, records, scores, manual_reviews)
    write_csv(metadata / "unreviewed_cross_label_candidates.csv", unreviewed_components, fields=list(scores[0]))
    info = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_master": str(MASTER),
        "output": str(OUTPUT),
        "source_manifest_sha256": snapshot["manifest_sha256"],
        "seed": SEED,
        "target_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
        "original_classes": len(json.loads((MASTER / "class_to_idx.json").read_text())),
        "retained_classes": len(mapping),
        "original_images": len(rows),
        "retained_images": len(included),
        "excluded_images": len(rows) - len(included),
        "exclusion_reason_counts": dict(reason_counts),
        "split_counts": dict(counts),
        "retained_groups": len({row["group_id"] for row in included}),
        "manual_review_pair_counts": dict(Counter(row["decision"] for row in manual_reviews)),
        "unreviewed_cross_label_candidate_components": len(unreviewed_components),
        "perceptual_screen": {"maximum_hamming_distance": 3, "automatic_same_label_maximum_mae": 1,
                              "automatic_same_label_maximum_rmse": 3,
                              "review_policy": "All strict cross-label matches, additional MAE<=8 or pHash-identical component pairs, and other-crop component pairs were visually reviewed. Lower-priority rice and dark-background leaf candidates remain unreviewed; similarity is not proof of identity."},
        "documented_leaf_images": sum(bool(row["documented_leaf_ids"]) for row in included),
        "traceable_images_without_documented_leaf_id": sum(not row["documented_leaf_ids"] for row in included),
        "training_only_images": sum(row["training_only"] for row in included),
        "exposure_record_sources": exposure_sources,
        "classes_missing_validation": [row["class_name"] for row in summaries if row["val"] == 0],
        "classes_missing_test": [row["class_name"] for row in summaries if row["test"] == 0],
        "classes_with_fewer_than_10_test_groups": sum(row["test_groups"] < 10 for row in summaries),
        "validation": {
            "all_retained_images_fully_decoded": True,
            "all_copies_sha256_verified": True,
            "no_related_groups_across_splits": True,
            "no_exact_decoded_rotation_flip_duplicates": True,
            "no_recorded_previous_exposure_in_holdouts": True,
            "no_offline_augmentations_in_holdouts": True,
        },
    }
    (staging / "dataset_info.json").write_text(json.dumps(info, indent=2))
    write_readme(staging, info)
    verify(workers, staging)
    staging.rename(OUTPUT)
    print("Published final dataset", OUTPUT, flush=True)


def write_readme(directory, info):
    readme = f'''# Final plant condition classification dataset

This release contains {info["retained_images"]:,} images across {info["retained_classes"]} plant-condition labels, prepared for training a custom CNN. Image files are copied byte for byte from the master snapshot. The class order is defined by `class_to_idx.json`.

| Split | Images | Purpose |
| --- | ---: | --- |
| train | {info["split_counts"].get("train", 0):,} | Fit weights and estimate any preprocessing statistics |
| val | {info["split_counts"].get("val", 0):,} | Select settings and checkpoints |
| test | {info["split_counts"].get("test", 0):,} | Evaluate after model and decision rules are fixed |

## Cleaning and grouping

Every master file underwent checksum verification and a full pixel decode of the first frame used by the image loader. Images with a side below 32 pixels, uniform images, corrupt files, invalid traceability records, groups with conflicting labels, and unresolved cross-label visual-review cases were excluded. Exact duplicate files and decoded copies identical under rotation or reflection retain one representative. Other distinct, consistently labelled views are retained together in one split.

Grouping combines master duplicate/augmentation groups, the available documented PlantVillage leaf IDs, conservative filename leaf-family hints, exact pixel identity, and reviewed or strict near-copy matches. Perceptual candidates require pixel comparison or review before grouping. Pinned upstream metadata and the review decisions are in `metadata/`.

The earlier project inventory also supplies documented grouping and visual-review evidence. Known comparison graphics, disease-name text, illustrations, and ambiguous/conflicting annotations are excluded wherever matching content is found. The earlier policy of excluding images without leaf IDs is not applied to this release.

The near-copy screen uses a rotation/reflection-aware perceptual hash with Hamming distance at most 3. Same-label candidates are grouped automatically only when their aligned 64-pixel thumbnails have mean absolute error at most 1 and root mean squared error at most 3 on the 0–255 scale. Cross-label matches are reviewed visually. All strict cross-label matches, additional component pairs with mean error at most 8 or identical perceptual hashes, and remaining pairs outside the common rice/dark-background-leaf families were reviewed. Lower-priority rice and dark-background leaf candidates remain unreviewed. Similarity alone does not establish that two photos show the same specimen. This screen can miss crops and other edits.

Every image has a stable SHA-256 image ID. {info["documented_leaf_images"]:,} retained images have matched documented leaf IDs. {info["traceable_images_without_documented_leaf_id"]:,} have traceable source records but lack documented leaf IDs. A content ID identifies an image and cannot establish biological independence. Filename families and near-copy grouping are inferred evidence. Additional unmarked related views or annotation errors may remain.

## Split policy

Groups are assigned with seed {SEED} and approximate 70/20/10 image targets within each class. Group sizes and fresh holdout availability take priority over exact proportions. Groups containing recorded images used in earlier experiments are confined to training. Recognized offline flip-augmentation families are also confined to training. The exposure snapshot is in `metadata/previous_exposures.json`.

There are {info["classes_with_fewer_than_10_test_groups"]} classes with fewer than 10 test groups. Per-class scores for rare labels require their sample counts and cautious interpretation. The complete group and image counts are in `split_summary.csv`.

## Using this release

The `train`, `val`, and `test` directories support an image-folder loader. Use `manifest.csv` rows with `status=included` for a manifest-based loader. Apply camera-orientation correction, select frame zero for multi-frame containers, convert to RGB, and use the same deterministic resize and scaling for inference and evaluation. Apply augmentation only during training.

`manifest.csv` accounts for every source image and records its final assignment or exclusion. `exclusions.csv` provides the omitted rows and reasons. The retained labels are inherited annotations and standardized source-folder labels. Source paths and labels are preserved for review and citation. Source-format verification cannot verify the biological diagnosis, scene relevance, or upstream reuse permissions.

## Reproducing the release

Use the project environment to run `python scripts/prepare_final_dataset.py audit`, then `group`, then `prepare`. Resolve any cross-label review candidates before preparation. Completed audits are cached for recovery. Existing releases are verified instead of overwritten. The master manifest fingerprint is `{info["source_manifest_sha256"]}`. Verify a release with `python scripts/prepare_final_dataset.py verify`.
'''
    (directory / "README.md").write_text(readme)


def verify(workers, root=OUTPUT):
    records = read_csv(root / "manifest.csv")
    mapping = json.loads((root / "class_to_idx.json").read_text())
    included = verify_records(records, mapping)
    expected_paths = {row["output_relative_path"] for row in included}
    actual_paths = {
        path.relative_to(root).as_posix()
        for split in ("train", "val", "test")
        for path in (root / split).rglob("*") if path.is_file()
    }
    assert actual_paths == expected_paths
    for split in ("train", "val", "test"):
        assert sorted(path.name for path in (root / split).iterdir() if path.is_dir()) == sorted(mapping)

    def verify_file(row):
        assert file_sha256(root / row["output_relative_path"]) == row["sha256"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(verify_file, included):
            pass
    info = json.loads((root / "dataset_info.json").read_text())
    assert info["retained_images"] == len(included)
    assert info["retained_classes"] == len(mapping)
    assert info["original_images"] == len(records)
    assert info["split_counts"] == dict(Counter(row["split"] for row in included))
    exposed = set(json.loads((root / "metadata/previous_exposures.json").read_text())["sha256"])
    assert not exposed.intersection(row["sha256"] for row in included if row["split"] != "train")
    print(f"Verified {len(included):,} images, {len(mapping)} classes, hashes, unique pixels, and disjoint groups.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["audit", "group", "prepare", "verify"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.phase == "audit":
        audit(args.workers)
    elif args.phase == "group":
        group(args.workers)
    elif args.phase == "prepare":
        prepare(args.workers)
    else:
        verify(args.workers)


if __name__ == "__main__":
    main()
