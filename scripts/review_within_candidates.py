"""Render the remaining within-split raw identity pairs with labels masked."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from review_integrity_candidates import ROOT, DATA, load_cases, thumbnail, align


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, nargs="+", help="Larger source views for uncertain case IDs")
    args = parser.parse_args()
    rows = load_cases()[366:]
    assert len(rows) == 426 and all(r["cross_split"] == "False" for r in rows)
    directory = ROOT / "runs/final177/additional_integrity_review/within"
    directory.mkdir(exist_ok=True)
    if args.cases:
        lookup = {int(r["case_id"]):r for r in rows}
        for case in args.cases:
            row = lookup[case]; pictures = []; sizes = []
            for key in ["first_final_path", "second_final_path"]:
                with Image.open(DATA / row[key]) as source:
                    source.seek(0); image = ImageOps.exif_transpose(source).convert("RGB")
                    sizes.append(image.size)
                    pictures.append(ImageOps.pad(image, (768,768), method=Image.Resampling.NEAREST, color=(128,128,128)))
            second, _ = align(pictures[0], pictures[1])
            page = Image.new("RGB", (1560,820), "white")
            ImageDraw.Draw(page).text((10,8), f"Identity case {case}; original sizes {sizes}; nearest fit768; second D4 aligned", fill="black")
            page.paste(pictures[0], (8,32)); page.paste(second, (784,32))
            page.save(directory / f"native_{case:06d}.jpg", quality=98)
        return
    records = []
    for offset in range(0, len(rows), 12):
        page = Image.new("RGB", (1428,1120), "white"); draw = ImageDraw.Draw(page)
        draw.text((12,8), "Within-split raw identity review; disease labels and model outputs masked", fill="black")
        name = f"within_{offset//12:03d}.jpg"
        for i,row in enumerate(rows[offset:offset+12]):
            first = thumbnail(DATA / row["first_final_path"])
            second, alignment = align(first, thumbnail(DATA / row["second_final_path"]))
            x = 12 + (i%3)*472; y = 34 + (i//3)*270
            page.paste(first,(x,y));page.paste(second,(x+230,y))
            draw.text((x,y+228),f"Case {row['case_id']} | {row['first_split']} / {row['second_split']} | distance {row['distance']}",fill="black")
            records.append({"case_id":int(row["case_id"]),"page":name,**alignment})
        target = directory/name;page.save(target,quality=94)
    (directory/"rendering.json").write_text(json.dumps({"cases_sha256":hashlib.sha256((directory.parent/"cases.csv").read_bytes()).hexdigest(),"count":426,"records":records,"scope":"Raw identity only; no automatic verdict or data/model changes."},indent=2))


if __name__ == "__main__":main()
