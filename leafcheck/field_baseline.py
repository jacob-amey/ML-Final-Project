"""Measure the frozen CNN on a separately screened development collection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .inference import Predictor
from .potato_policy import distribution
from .reliability_study import ece, measures
from .upload_study import ARTIFACT, member_rows, pixels, read, score, summarize


def run(release_dir, output):
    if (output / "baseline.json").exists() or (output / "predictions.csv").exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    model = Predictor(ARTIFACT, "cpu")
    release = read(release_dir / "release.json")
    hashes = release["files_sha256"]
    if (
        release["base_release"] != model.metadata["dataset_fingerprint"]
        or any(sha256(release_dir / path) != value for path, value in hashes.items())
        or hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() != release["fingerprint"]
        or "val_only_counts" not in release
    ):
        raise ValueError("Expected a pinned development-only release")
    classes = model.classes
    mapping = {label: index for index, label in enumerate(classes)}
    rows = member_rows(release_dir, mapping)
    if (len(rows) != sum(release["val_only_counts"].values())
            or len({r["image_id"] for r in rows}) != len(rows)
            or any(row["split"] != "val" for row in rows)):
        raise ValueError("Invalid development membership")

    images = pixels(rows, model.spec["input_size"])
    probability = distribution(model.scoring_model, images, model.spec, model.device, size=64)
    truth = np.asarray([int(row["class_id"]) for row in rows])
    guess = probability.argmax(1)
    confidence = probability.max(1)
    slots = np.unique(truth)
    scores = score(truth, guess, classes, slots)
    per_class = [scores["per_class"][i] for i in slots]
    result = summarize(scores)
    result.update(
        crop_accuracy=1 - scores["wrong_crop"] / len(rows),
        balanced_accuracy=float(np.mean([row["recall"] for row in per_class])),
        macro_precision=float(np.mean([row["precision"] for row in per_class])),
        weighted_f1=sum(row["f1"] * row["support"] for row in per_class) / len(rows),
        per_class=per_class,
        blight=[row for row in scores["blight"] if row["total"]],
        largest_errors=scores["largest_errors"],
        confidence_ece_15=ece(confidence, truth == guess),
        high_score_incorrect=int(np.count_nonzero((confidence >= .9) & (truth != guess))),
    )
    healthy = np.array([label.endswith("__healthy") for label in classes])
    result["healthy_reference"] = int(healthy[truth].sum())
    result["healthy_false_condition"] = int(np.count_nonzero(healthy[truth] & ~healthy[guess]))

    policy = read(ROOT / "leafcheck-demo/backend/reliability_policy.json")
    if policy["checkpoint_sha256"] != model.metadata["checkpoint_sha256"] or policy["inference_policy"] != model.policy["name"]:
        raise ValueError("Displayed policy differs from the audited model")
    accept = confidence >= policy["threshold"]
    result["already_deployed_gate"] = {"threshold": policy["threshold"], **measures(truth == guess, accept)}
    output.mkdir(parents=True, exist_ok=True)
    records = [dict(image_id=row["image_id"], group_id=row["group_id"], reference=row["class_name"],
                    predicted=classes[guess[i]], score=float(confidence[i]), shown=bool(accept[i]))
               for i, row in enumerate(rows)]
    write_csv(output / "predictions.csv", records)
    receipt = {
        "created_utc": utc_now(), "checkpoint": model.metadata["checkpoint_sha256"],
        "scoring": model.policy["name"], "class_count": len(classes), "release": release["fingerprint"],
        "group_count": len({row["group_id"] for row in rows}), "results": result,
        "predictions_sha256": sha256(output / "predictions.csv"), "test_opened": False,
    }
    atomic_json(output / "baseline.json", receipt)
    print(summarize(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.release, args.output)
