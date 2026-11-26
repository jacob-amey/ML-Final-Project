"""Score the frozen single-label Pretoria corn development cohort."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .inference import Predictor
from .upload_study import ARTIFACT, pixels, probabilities, read, rows, score, summarize


PLAN = ROOT / "configs/leafcheck_pretoria_baseline_v1.json"
RAW = ROOT / "data/raw/leafcheck_field_v9/pretoria"


def run(output):
    if (output / "metrics.json").exists() or (output / "predictions.csv").exists():
        raise FileExistsError(output)
    plan = read(PLAN)
    manifest = ROOT / plan["manifest"]
    if sha256(manifest) != plan["manifest_sha256"]:
        raise ValueError("The registered development manifest changed")
    examples = [dict(row, path=RAW / f"{int(row['source_id'])}.jpg") for row in rows(manifest)
                if row["screen"] == "no_candidate"]
    if len(examples) != plan["development_count"] or len({r["sha256"] for r in examples}) != len(examples):
        raise ValueError("Unexpected development membership")
    torch.set_num_threads(4)
    predictor = Predictor(ARTIFACT, "cpu")
    for key in ("checkpoint_sha256", "dataset_fingerprint"):
        if predictor.metadata[key] != plan[key]:
            raise ValueError(f"Unexpected model identity {key}")
    if predictor.policy["name"] != plan["policy"]:
        raise ValueError("Unexpected deployed scoring policy")
    classes = predictor.classes
    focus = np.array([i for i, label in enumerate(classes) if label.startswith("corn__")])
    labels = np.array([classes.index(r["label"]) for r in examples])
    sources = np.array([r["camera_date"].split()[0] for r in examples])
    p = probabilities(predictor.scoring_model, pixels(examples, predictor.spec["input_size"]),
                      predictor.spec, predictor.device, batch_size=32)
    within_corn = p[:, focus]
    within_corn = within_corn / within_corn.sum(1, keepdims=True)
    selected = {"automatic": (p.argmax(1), p.max(1)),
                "supplied_corn": (focus[within_corn.argmax(1)], within_corn.max(1))}
    represented = np.unique(labels)
    measurements = {}
    for method, (predictions, _) in selected.items():
        result = score(labels, predictions, classes, represented)
        day_counts = {}
        for day in np.unique(sources):
            day_rows = sources == day
            day_counts[day] = {"count": int(day_rows.sum()),
                               "correct": int(((predictions == labels) & day_rows).sum()),
                               "predicted_corn": int((np.isin(predictions, focus) & day_rows).sum())}
        measurements[method] = dict(summarize(result), largest_errors=result["largest_errors"],
                                    per_class=[result["per_class"][i] for i in represented], capture_days=day_counts)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "predictions.csv"
    write_csv(path, [dict(source_id=r["source_id"], original_sha256=r["sha256"], truth=r["label"],
                         capture_day=sources[i], **{f"{method}_label": classes[predicted[i]]
                                                   for method, (predicted, _) in selected.items()},
                         **{f"{method}_score": float(scores[i]) for method, (_, scores) in selected.items()})
                     for i, r in enumerate(examples)])
    atomic_json(output / "metrics.json", {
        "protocol_sha256": sha256(PLAN), "checkpoint_sha256": plan["checkpoint_sha256"],
        "manifest_sha256": plan["manifest_sha256"], "predictions_sha256": sha256(path),
        "automatic_eligible_classes": len(classes), "conditional_classes": [classes[i] for i in focus],
        "macro_average_classes": [classes[i] for i in represented], "measurements": measurements,
        "test_used": False, "app_changed": False, "completed_utc": utc_now(),
    })
    for name, measurement in measurements.items():
        print(name, summarize(measurement), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
