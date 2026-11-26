"""Replay the frozen model on the fixed unmarked rice field sample."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .inference import Predictor
from .upload_study import ARTIFACT, read


MEMBERSHIP = ROOT / "data/manifests/leafcheck_rice_field_preflight.json"
RAW = ROOT / "data/raw/leafcheck_field_v12/rice-intake"
BLAST_FAMILY = {"rice__blast", "rice__leaf_blast"}


def diagnostic(output):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("Preserving the existing diagnostic")
    torch.set_num_threads(4)
    model = Predictor(ARTIFACT)
    freeze = read(ARTIFACT / "freeze.json")
    manifest = read(MEMBERSHIP)
    if (freeze["checkpoint_sha256"] != model.metadata["checkpoint_sha256"]
            or manifest["master_fingerprint"] != model.metadata["dataset_fingerprint"]
            or manifest["roles_assigned"] or model.policy["name"] != "d4_probability_mean"):
        raise ValueError("Unexpected frozen deployment or intake")
    rows = [row for row in manifest["records"] if row["screen"] == "unassigned"]
    if len(rows) != len({row["sha256"] for row in rows}) or not rows:
        raise ValueError("The fixed sample is incomplete or contains repeated bytes")
    rice = [name for name in model.classes if name.startswith("rice__")]
    if not BLAST_FAMILY.issubset(rice):
        raise ValueError("Expected two leaf-oriented blast outputs")

    results = []
    for row in rows:
        path = RAW / (row["sha256"] + ".jpg")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Changed source original at {row['archive_member']}")
        prediction = model.predict(path)
        scores = prediction["class_scores"]
        accepted = BLAST_FAMILY if row["folder_label"] == "Blast" else {row["label_hypothesis"]}
        if not accepted.issubset(rice):
            raise ValueError(f"Unsupported label in the registered diagnostic: {accepted}")
        predicted = prediction["label"]
        conditional = max(rice, key=scores.__getitem__)
        results.append({"member": row["archive_member"], "sha256": row["sha256"], "filename_day": row["filename_day"],
                        "label": row["folder_label"], "acceptable_classes": ";".join(sorted(accepted)),
                        "automatic_label": predicted, "automatic_score": prediction["score"],
                        "automatic_correct": predicted in accepted, "automatic_wrong_crop": predicted not in rice,
                        "given_rice_label": conditional, "given_rice_raw_score": scores[conditional],
                        "given_rice_correct": conditional in accepted})
    by_label, by_day = defaultdict(list), defaultdict(list)
    for row in results:
        by_label[row["label"]].append(row)
        by_day[row["filename_day"]].append(row)
    tracked = ("automatic_correct", "automatic_wrong_crop", "given_rice_correct")

    def counts(rows):
        return {"support": len(rows), **{key: sum(row[key] for row in rows) for key in tracked},
                "automatic_labels": Counter(row["automatic_label"] for row in rows),
                "given_rice_labels": Counter(row["given_rice_label"] for row in rows)}

    predictions = output / "predictions.csv"
    write_csv(predictions, results)
    evidence = {"measured_utc": utc_now(), "sample_sha256": sha256(MEMBERSHIP),
                "checkpoint_sha256": model.metadata["checkpoint_sha256"], "predictions_sha256": sha256(predictions),
                "automatically_eligible_classes": len(model.classes), "rice_classes": rice,
                "blast_family": sorted(BLAST_FAMILY), "by_condition": {key: counts(data) for key, data in by_label.items()},
                "by_filename_day": {key: counts(data) for key, data in sorted(by_day.items())},
                "total": counts(results), "course_test_opened": False, "deployment_changed": False}
    atomic_json(output / "metrics.json", evidence)
    print("Total", {key: evidence["total"][key] for key in ("support", *tracked)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    diagnostic(parser.parse_args().output)
