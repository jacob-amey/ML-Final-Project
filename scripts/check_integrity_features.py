"""Raw-image correspondence checks for integrity, never CNN performance.

Run with the isolated integrity_env. OpenCV SIFT/RANSAC follows the documented
API: https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
Results flag evidence to inspect; they do not automatically change any label.
"""

import csv
from io import BytesIO
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from review_integrity_candidates import load_cases

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "runs/final177/additional_integrity_review"
DATA = ROOT / "data/processed/leafcheck177_final"


def read(path):
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((1024,1024), Image.Resampling.LANCZOS)
        return np.array(image)


def match(first, second, name):
    detector = cv2.SIFT_create(nfeatures=6000, contrastThreshold=.008, edgeThreshold=20)
    kp1, desc1 = detector.detectAndCompute(cv2.cvtColor(first, cv2.COLOR_RGB2GRAY), None)
    candidates = []
    for flipped in [False, True]:
        image = np.ascontiguousarray(second[:, ::-1] if flipped else second)
        kp2, desc2 = detector.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), None)
        good = []
        if desc1 is not None and desc2 is not None and len(desc2) >= 2:
            for pair in cv2.BFMatcher(cv2.NORM_L2).knnMatch(desc1, desc2, k=2):
                if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance: good.append(pair[0])
        H = None; mask = None; hull_fraction = 0.0; destination_fraction = 0.0; unique_destinations = 0
        if len(good) >= 4:
            left = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            right = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            H, mask = cv2.findHomography(left, right, cv2.RANSAC, 2.5)
            if mask is not None and int(mask.sum()) >= 3:
                chosen = mask.ravel().astype(bool)
                hull_fraction = cv2.contourArea(cv2.convexHull(left[chosen])) / (first.shape[0]*first.shape[1])
                destination_fraction = cv2.contourArea(cv2.convexHull(right[chosen])) / (image.shape[0]*image.shape[1])
                unique_destinations = len(np.unique(right[chosen].reshape(-1,2).round(1),axis=0))
        inliers = int(mask.sum()) if mask is not None else 0
        rank = int(np.linalg.matrix_rank(H)) if H is not None else 0
        distributed_geometry = inliers >= 8 and unique_destinations >= 6 and min(hull_fraction,destination_fraction) > .001 and rank == 3
        record = {"mirrored_second":flipped, "keypoints_first":len(kp1), "keypoints_second":len(kp2),
                  "ratio_test_matches":len(good), "ransac_inliers":inliers, "inlier_hull_fraction_of_first":hull_fraction,
                  "inlier_hull_fraction_of_second":destination_fraction,"unique_inlier_destination_positions":unique_destinations,
                  "homography_rank":rank,"distributed_geometry_for_visual_review":bool(distributed_geometry),
                  "homography":H.tolist() if H is not None else None}
        candidates.append((record, image, kp2, good, mask))
    record, image, kp2, good, mask = max(candidates, key=lambda item:(item[0]["distributed_geometry_for_visual_review"], item[0]["ransac_inliers"], item[0]["ratio_test_matches"]))
    visual = cv2.drawMatches(first, kp1, image, kp2, good, None,
                             matchesMask=mask.ravel().tolist() if mask is not None else None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    Image.fromarray(visual).save(DIRECTORY / f"features_{name}.jpg", quality=95)
    return {"case":name, "attempts":[item[0] for item in candidates], "display":f"features_{name}.jpg",
            "interpretation":"Geometric correspondences require visual confirmation. Few/collinear matches do not establish identity; no match does not prove different biological specimens or every crop is unrelated."}


def main():
    cv2.setNumThreads(2); cv2.setRNGSeed(434)
    rows = {int(row["case_id"]):row for row in load_cases()}
    results = []
    for case in [284,286,351,477,525,556,482,102]:
        row = rows[case]
        result = match(read(DATA/row["first_final_path"]), read(DATA/row["second_final_path"]), str(case))
        results.append(result)
        print(case, [(r["ratio_test_matches"],r["ransac_inliers"],round(r["inlier_hull_fraction_of_first"],4)) for r in result["attempts"]], flush=True)
    # Known same-photo software control: resize/crop/JPEG from one raw image.
    source = read(DATA/rows[482]["first_final_path"])
    image = Image.fromarray(source)
    w,h = image.size; image = image.crop((int(w*.1),int(h*.1),int(w*.9),int(h*.9))).resize((256,256),Image.Resampling.LANCZOS)
    buffer=BytesIO(); image.save(buffer,"JPEG",quality=85); buffer.seek(0)
    with Image.open(buffer) as generated: transformed=np.array(generated.convert("RGB"))
    results.append(match(source, transformed, "same_photo_software_control"))
    output = {"software":{"opencv":cv2.__version__,"numpy":np.__version__},
              "scope":"Raw-file identity review only; no CNN outputs, clinical labels, test performance or independent evaluation images.",
              "positive_control":"Same photograph cropped to its central80%, resized256 and JPEG quality85, used only to check correspondence software.",
              "results":results}
    (DIRECTORY / "feature_matching_review.json").write_text(json.dumps(output,indent=2))


if __name__ == "__main__": main()
