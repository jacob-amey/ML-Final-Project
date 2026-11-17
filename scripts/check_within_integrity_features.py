"""Supplement raw within-split identity review with correspondence evidence."""
from io import BytesIO
import json
from PIL import Image
import check_integrity_features as checks


def main():
    checks.DIRECTORY = checks.DIRECTORY / "within"
    checks.cv2.setNumThreads(2); checks.cv2.setRNGSeed(434)
    lookup = {int(r["case_id"]):r for r in checks.load_cases()}
    cases = [327,226,669,710,648,444,342,344,671,331,381,780,185,352,129]
    records = []
    for case in cases:
        row = lookup[case]; assert row["cross_split"] == "False"
        result = checks.match(checks.read(checks.DATA/row["first_final_path"]), checks.read(checks.DATA/row["second_final_path"]), str(case))
        records.append(result)
        print(case, [(r["ratio_test_matches"],r["ransac_inliers"],r["distributed_geometry_for_visual_review"]) for r in result["attempts"]], flush=True)
    # Same train-photo software control as the prior cross-split audit.
    source = checks.read(checks.DATA / lookup[482]["first_final_path"])
    image = Image.fromarray(source); w,h = image.size
    image = image.crop((int(w*.1),int(h*.1),int(w*.9),int(h*.9))).resize((256,256),Image.Resampling.LANCZOS)
    buffer = BytesIO(); image.save(buffer,"JPEG",quality=85); buffer.seek(0)
    with Image.open(buffer) as result:
        transformed = checks.np.array(result.convert("RGB"))
    records.append(checks.match(source,transformed,"same_photo_software_control"))
    (checks.DIRECTORY/"feature_matching_review.json").write_text(json.dumps({"software":{"opencv":checks.cv2.__version__,"numpy":checks.np.__version__},"scope":"Raw identity only; no CNN predictions, clinical labels or performance. Flags require manual review; a failed match does not prove biological independence.","results":records},indent=2))


if __name__ == "__main__":main()
