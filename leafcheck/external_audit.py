"""Audit the existing PlantDoc external collection without computing predictions."""

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageOps
from .common import ROOT, DATA, RUNS, atomic_json, sha256, write_csv
from scripts.prepare_final_dataset import canonical_pixel_hash, perceptual_hashes, dihedral_views

OUTPUT = RUNS / "external_audit"
MASTER = ROOT / "Plant_Dataset_Updated_2026-11-11"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    inventory = list(csv.DictReader((ROOT / "data/manifests/image_inventory.csv").open()))
    external = [row for row in inventory if row["status"] == "included" and row["split"] == "external_test"]
    master = list(csv.DictReader((DATA / "manifest.csv").open()))
    audits = [json.loads(line) for line in (DATA / "metadata/image_audit.jsonl").open()]
    raw = {row["sha256"] for row in master}; pixels = {row["pixel_d4_sha256"] for row in master}
    prior_groups = {identifier for row in master for identifier in row["prior_review_group_ids"].split("|") if identifier}
    buckets = defaultdict(set)
    for index, audit in enumerate(audits):
        if not audit.get("phashes"): continue
        value = audit["phashes"][0]
        for part in range(4): buckets[part, (value >> (part * 16)) & 65535].add(index)
    def inspect(row):
        path = ROOT / row["image_path"]; assert sha256(path) == row["sha256"]
        with Image.open(path) as source:
            source.seek(0); source.load(); image = ImageOps.exif_transpose(source).convert("RGB")
            return canonical_pixel_hash(image), perceptual_hashes(image), np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), np.float32)
    results = []; candidates = []
    labels = {"early_blight": "tomato__early_blight", "healthy": "tomato__healthy", "late_blight": "tomato__late_blight"}
    review_path = OUTPUT / "manual_review.csv"
    if not review_path.exists():
        review_path = ROOT / "data/manifests/leafcheck177_external_review.csv"
    reviewed = {(row["external_path"], row["master_path"]): row for row in csv.DictReader(review_path.open())} if review_path.exists() else {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row, (pixel_hash, hashes, thumbnail) in zip(external, pool.map(inspect, external)):
            reasons = []
            group = "prior_inventory:" + row["source"] + ":" + row["source_version"] + ":" + row["group_id"]
            if row["sha256"] in raw: reasons.append("raw_hash_in_master")
            if pixel_hash in pixels: reasons.append("decoded_rotation_flip_match_in_master")
            if group in prior_groups: reasons.append("previously_reviewed_source_group_in_master")
            possible = set()
            if not reasons:
                for value in hashes:
                    for part in range(4): possible.update(buckets.get((part, (value >> (part * 16)) & 65535), set()))
                by_group = {}
                for index in possible:
                    audit = audits[index]
                    distance = min((value ^ audit["phashes"][0]).bit_count() for value in hashes)
                    if distance > 3: continue
                    with Image.open(MASTER / audit["master_path"]) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
                        second = np.asarray(image, np.float32)
                    mae = min(float(np.abs(thumbnail - view).mean()) for view in dihedral_views(second))
                    root_key = audit["pixel_d4_sha256"]
                    candidate = {"external_path": row["image_path"], "external_class": labels[row["project_label"]],
                                 "master_path": audit["master_path"], "distance": distance, "mean_absolute_error": round(mae, 6)}
                    if root_key not in by_group or mae < by_group[root_key]["mean_absolute_error"]:
                        by_group[root_key] = candidate
                for candidate in sorted(by_group.values(), key=lambda r:r["mean_absolute_error"]):
                    decision = reviewed.get((candidate["external_path"], candidate["master_path"]))
                    if decision and decision["decision"] in {"same_source", "uncertain"}:
                        reasons.append("reviewed_related_or_uncertain_master_match")
                    elif not decision:
                        candidates.append(candidate)
            results.append({"image_id": "sha256:" + row["sha256"], "image_path": row["image_path"],
                            "class_name": labels[row["project_label"]], "source": row["source"], "source_version": row["source_version"],
                            "source_path": row["source_path"], "group_id": group, "sha256": row["sha256"],
                            "pixel_d4_sha256": pixel_hash, "status": "excluded" if reasons else "included",
                            "exclusion_reasons": "|".join(sorted(set(reasons)))})
    write_csv(OUTPUT / "candidates.csv", candidates, fields=["external_path", "external_class", "master_path", "distance", "mean_absolute_error"])
    write_csv(OUTPUT / "manifest.csv", results)
    summary = {"original_images": len(external), "retained_images_before_unresolved_reviews": sum(r["status"] == "included" for r in results),
               "pending_pairs": len(candidates), "excluded_images": sum(r["status"] == "excluded" for r in results),
               "exclusion_counts": dict(Counter(reason for r in results for reason in r["exclusion_reasons"].split("|") if reason)),
               "test_predictions_computed": False}
    atomic_json(OUTPUT / "audit.json", summary); print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
