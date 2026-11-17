"""Render remaining cross-split identity candidates without predictions or labels.

This is a raw-file integrity review aid, not a model-development visualization.
It never edits image data, labels, groups or split membership.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/leafcheck177_final"


def load_cases():
    """Reconstruct the ordered review list from the unchanged release metadata."""
    directory = ROOT / "runs/final177/additional_integrity_review"
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "cases.csv"
    with (DATA / "manifest.csv").open() as file:
        included = {row["master_relative_path"]:row for row in csv.DictReader(file) if row["status"] == "included"}
    with (DATA / "metadata/unreviewed_cross_label_candidates.csv").open() as file:
        candidates = list(csv.DictReader(file))
    rows = []
    for index, row in enumerate(candidates):
        first = included[row["first"]]; second = included[row["second"]]
        rows.append({"case_id":str(index), **row, "first_final_path":first["output_relative_path"],
                     "second_final_path":second["output_relative_path"], "first_split":first["split"],
                     "second_split":second["split"], "cross_split":str(first["split"] != second["split"]),
                     "touches_test":str("test" in [first["split"],second["split"]])})
    rows.sort(key=lambda row:(row["cross_split"] != "True", float(row["mean_absolute_error"]), int(row["case_id"])))
    assert len(rows) == 792
    if source.exists():
        assert rows == list(csv.DictReader(source.open())), "Review list changed after the recorded integrity decisions"
    else:
        with source.open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def thumbnail(path):
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source).convert("RGB")
        return ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR, color=(128, 128, 128))


def align(first, second):
    left = np.asarray(first.resize((64, 64)), dtype=np.int16)
    options = []
    for flip in [False, True]:
        image = ImageOps.mirror(second) if flip else second
        for angle in [0, 90, 180, 270]:
            view = image.rotate(angle)
            score = float(np.abs(left - np.asarray(view.resize((64, 64)), dtype=np.int16)).mean())
            options.append((score, flip, angle, view))
    score, flip, angle, view = min(options, key=lambda item:item[0])
    return view, {"second_mirrored":flip, "second_rotation_degrees":angle, "preview_aligned_mae":score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=36, help="Skip the first36 already reviewed pairs")
    parser.add_argument("--stop", type=int, default=366, help="All366 cross-split candidates, excluding within-split candidates")
    parser.add_argument("--native-cases", type=int, nargs="+", help="Larger source-image views for selected case IDs; original dimensions are recorded")
    args = parser.parse_args()
    directory = ROOT / "runs/final177/additional_integrity_review"
    source = directory / "cases.csv"
    rows = load_cases()
    assert sum(row["cross_split"] == "True" for row in rows) == 366
    assert all(row["cross_split"] == "True" for row in rows[:366])
    assert 0 <= args.start < args.stop <= 366
    if args.native_cases:
        records = []
        by_id = {int(row["case_id"]):row for row in rows[:366]}
        for case in args.native_cases:
            row = by_id[case]; images = []; sizes = []
            for key in ["first_final_path", "second_final_path"]:
                with Image.open(DATA / row[key]) as source_image:
                    source_image.seek(0)
                    image = ImageOps.exif_transpose(source_image).convert("RGB")
                    sizes.append(image.size)
                    images.append(ImageOps.pad(image, (768,768), method=Image.Resampling.NEAREST, color=(128,128,128)))
            second, alignment = align(images[0], images[1])
            page = Image.new("RGB", (1560,820), "white"); draw = ImageDraw.Draw(page)
            page.paste(images[0], (8,32)); page.paste(second, (784,32))
            draw.text((10,8), f"Identity case {case}; original dimensions {sizes}; nearest-neighbor fit768 for display; second D4 aligned", fill="black")
            target = directory / f"native_case_{case:06d}.jpg"; page.save(target, quality=98)
            records.append({"case_id":case, "original_dimensions":sizes, **alignment,
                            "file":target.name, "sha256":hashlib.sha256(target.read_bytes()).hexdigest()})
            print(target, flush=True)
        (directory / "native_review_rendering.json").write_text(json.dumps(records, indent=2))
        return
    records = []
    for offset in range(args.start, args.stop, 12):
        selected = rows[offset:min(offset+12, args.stop)]
        page = Image.new("RGB", (1428, 1120), "white"); draw = ImageDraw.Draw(page)
        draw.text((12, 8), "Raw identity review only — no model predictions or disease-label display", fill="black")
        for i, row in enumerate(selected):
            first = thumbnail(DATA / row["first_final_path"])
            second, alignment = align(first, thumbnail(DATA / row["second_final_path"]))
            x = 12 + (i % 3) * 472; y = 34 + (i // 3) * 270
            page.paste(first, (x, y)); page.paste(second, (x+230, y))
            draw.text((x, y+228), f"Case {row['case_id']} | {row['first_split']} / {row['second_split']} | hash distance {row['distance']}", fill="black")
            draw.text((x, y+244), f"Second view aligned: mirror={alignment['second_mirrored']}, rotation={alignment['second_rotation_degrees']}", fill="black")
            records.append({"case_id":int(row["case_id"]), "ordered_index":offset+i,
                            "page":f"cross_page_{offset//12:03d}.jpg", **alignment})
        target = directory / f"cross_page_{offset//12:03d}.jpg"
        page.save(target, quality=94)
        print(target, flush=True)
    (directory / "cross_review_rendering.json").write_text(json.dumps({
        "cases_sha256":hashlib.sha256(source.read_bytes()).hexdigest(), "start":args.start, "stop":args.stop,
        "scope":"Raw-file identity only; no automatic verdict, label changes, model outputs or test performance.",
        "records":records}, indent=2))


if __name__ == "__main__": main()
