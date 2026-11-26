"""Screen a frozen CNN feature probe with whole sources and categories held out."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import ROOT, atomic_json, sha256, utc_now, write_csv
from .field_expansion import load_inputs
from .inference import Predictor
from .preprocessing import prepare_batch
from .unsupported_screen import DATA, MANIFEST, RAW, auroc
from .upload_study import ARTIFACT, pixels, read, rows


PLAN = ROOT / "configs/leafcheck_upload_support_probe_v1.json"


@torch.inference_mode()
def representations(predictor, raw):
    vectors, maximum_scores = [], []
    for batch in raw.split(48):
        x = prepare_batch(batch, predictor.spec, predictor.device)
        pooled, probabilities = [], []
        for reflected in (x, x.flip(-1)):
            for turns in range(4):
                view = torch.rot90(reflected, turns, (-2, -1))
                feature_map = predictor.model.features(view.contiguous(memory_format=torch.channels_last))
                vectors_per_view = predictor.model.classifier[0](feature_map).flatten(1)
                pooled.append(vectors_per_view)
                probabilities.append(predictor.model.classifier[-1](vectors_per_view).softmax(1))
        vectors.append(torch.stack(pooled).mean(0).cpu())
        maximum_scores.append(torch.stack(probabilities).mean(0).amax(1).cpu())
    return torch.cat(vectors).double(), torch.cat(maximum_scores).numpy()


def fit(positive_sources, negative_by_type, l2):
    sources = [*positive_sources, *(group for groups in negative_by_type for group in groups)]
    masses = [0.5 / len(positive_sources)] * len(positive_sources)
    for groups in negative_by_type:
        masses.extend([0.5 / len(negative_by_type) / len(groups)] * len(groups))
    x = torch.cat(sources)
    sample_weight = torch.cat([
        torch.full((len(source),), mass / len(source), dtype=x.dtype)
        for source, mass in zip(sources, masses, strict=True)
    ])
    positive_count = sum(len(source) for source in positive_sources)
    y = (torch.arange(len(x)) < positive_count).to(x.dtype)
    center = (sample_weight[:, None] * x).sum(0)
    scale = ((sample_weight[:, None] * (x - center).square()).sum(0)).sqrt().clamp_min(1e-8)
    z = (x - center) / scale
    linear = torch.nn.Linear(x.shape[1], 1, dtype=x.dtype)
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
    optimizer = torch.optim.LBFGS(linear.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        point_loss = F.binary_cross_entropy_with_logits(linear(z).flatten(), y, reduction="none")
        loss = (point_loss * sample_weight).sum() + l2 * linear.weight.square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    gradient = torch.cat([parameter.grad.flatten() for parameter in linear.parameters()])
    if not all(torch.isfinite(parameter).all() for parameter in linear.parameters()) or gradient.norm() > 0.001:
        raise RuntimeError("The support probe did not converge to a finite fit")

    @torch.inference_mode()
    def evaluate(features):
        return linear((features - center) / scale).flatten().numpy()

    return evaluate


def run(output):
    if (output / "metrics.json").exists() or (output / "predictions.csv").exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    plan = read(PLAN)
    predictor = Predictor(ARTIFACT, "cpu")
    if predictor.metadata["checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("Unexpected frozen weights")
    lineage = {"initial_weights": plan["checkpoint_sha256"]}
    _, field, previous, additional = load_inputs(DATA / "leafcheck_field_v2_sampled", lineage)
    _, pakistan_release, _, pakistan_members = load_inputs(DATA / "leafcheck_pakistan_tomato_dev_v1", lineage)
    size = predictor.spec["input_size"]
    development = {
        fold["supported"]: [row for row in previous + additional
                            if row["split"] == "val" and row["source"] == fold["supported"]]
        for fold in plan["folds"] if fold["supported"] != "pakistan"
    }
    development["pakistan"] = pakistan_members
    positives = {
        name: representations(predictor, pixels(members, size))
        for name, members in development.items()
    }
    records = [r for r in rows(MANIFEST) if r["status"] == "included"]
    if any(r["split"] != "val" or sha256(RAW / (r["image_id"] + ".jpg")) != r["sha256"] for r in records):
        raise ValueError("Unexpected unsupported development bytes or split")
    types = ("nonplant", "unsupported_plant")
    negatives = {}
    for fold in plan["folds"]:
        for group in types:
            category = fold[group]
            subset = [dict(r, path=RAW / (r["image_id"] + ".jpg"))
                      for r in records if r["group"] == group and r["category"] == category]
            features, scores = representations(predictor, pixels(subset, size))
            negatives[category] = (features, scores, [r["image_id"] for r in subset])
    folds, predictions = [], []
    for fold in plan["folds"]:
        source = fold["supported"]
        classifier = fit(
            [features for name, (features, _) in positives.items() if name != source],
            [[negatives[other[group]][0] for other in plan["folds"] if other[group] != fold[group]]
             for group in types],
            plan["l2"],
        )
        x, baseline_positive = positives[source]
        probe_positive = classifier(x)
        result = {"held_out": fold, "supported_count": len(x), "metrics": {}}
        predictions.extend({"image_id": row["image_id"], "category": "", "supported_source": source,
                            "group": "supported", "max_softmax": base_value, "feature_logit": probe_value}
                           for row, base_value, probe_value
                           in zip(development[source], baseline_positive, probe_positive, strict=True))
        for group in types:
            x_negative, baseline_negative, image_ids = negatives[fold[group]]
            probe_negative = classifier(x_negative)
            base = auroc(baseline_positive, baseline_negative)
            candidate = auroc(probe_positive, probe_negative)
            result["metrics"][group] = {
                "count": len(x_negative), "max_softmax_auroc": base,
                "feature_auroc": candidate, "gain": candidate - base,
            }
            predictions.extend({"image_id": image_id, "category": fold[group], "supported_source": source,
                                "group": group, "max_softmax": baseline_item, "feature_logit": probe_item}
                               for image_id, baseline_item, probe_item
                               in zip(image_ids, baseline_negative, probe_negative, strict=True))
        folds.append(result)
    summary = {group: {
        key: float(np.mean([fold["metrics"][group][key] for fold in folds]))
        for key in ("max_softmax_auroc", "feature_auroc", "gain")
    } for group in types}
    passed = all(values["feature_auroc"] >= plan["min_macro_auroc_each_type"]
                 and values["gain"] >= plan["min_macro_auroc_gain_each_type"] for values in summary.values())
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions.csv", predictions)
    atomic_json(output / "metrics.json", {
        "evaluated_utc": utc_now(), "plan_sha256": sha256(PLAN), "checkpoint_sha256": plan["checkpoint_sha256"],
        "negative_manifest_sha256": sha256(MANIFEST), "predictions_sha256": sha256(output / "predictions.csv"),
        "supported_fingerprints": [field["fingerprint"], pakistan_release["fingerprint"]],
        "folds": folds, "macro_by_unsupported_type": summary, "passed_development_gate": passed,
        "threshold_fit": False, "deployment_changed": False, "heldout_used": False,
    })
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
