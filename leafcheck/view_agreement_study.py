"""Measure whether D4 view agreement ranks unreliable responses on development data."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, atomic_json, sha256, utc_now
from .evaluate_feature_style import development_pixels, development_rows
from .inference import Predictor
from .preprocessing import prepare_batch
from .unsupported_screen import ARTIFACT, MANIFEST, RAW, auroc, pixels, read, rows


PLAN = ROOT / "configs/leafcheck_view_agreement_v1.json"
REFERENCE = "max_softmax"
CANDIDATE = "negative_disagreement"
MIN_PROBABILITY = torch.finfo(torch.float32).tiny


@torch.inference_mode()
def measurements(predictor, raw):
    scores = []
    predictions = []
    checked = False
    for batch in raw.split(32):
        x = prepare_batch(batch, predictor.spec, predictor.device)
        distributions = torch.stack([
            predictor.model(torch.rot90(view, n, (-2, -1)).contiguous(memory_format=torch.channels_last)).softmax(1)
            for view in (x, x.flip(-1)) for n in range(4)
        ])
        p = distributions.mean(0)
        if not checked:
            deployed = predictor.scoring_model(x[:1]).softmax(1)
            torch.testing.assert_close(p[:1], deployed, atol=1e-5, rtol=1e-5)
            if p[:1].argmax().item() != deployed.argmax().item():
                raise ValueError("View average changed the deployed label")
            checked = True
        confidence, top = p.max(1)
        log_ratio = distributions.clamp_min(MIN_PROBABILITY).log() - p.clamp_min(MIN_PROBABILITY).log()
        divergence = (distributions * log_ratio).sum(2).mean(0)
        match = (distributions.argmax(2) == top).float().mean(0)
        scores.append(torch.stack((confidence, -divergence.clamp_min(0), match), dim=1).cpu())
        predictions.append(top.cpu())
    signals = torch.cat(scores).numpy().T
    return dict(zip((REFERENCE, CANDIDATE, "top_view_agreement"), signals, strict=True)), torch.cat(predictions).numpy()


def correctness_auc(signal, correct):
    return {key: auroc(value[correct], value[~correct]) if correct.any() and (~correct).any() else None
            for key, value in signal.items()}


def run(output):
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    plan = read(PLAN)
    predictor = Predictor(ARTIFACT, "cpu")
    if plan["checkpoint_sha256"] != predictor.metadata["checkpoint_sha256"] or predictor.policy["name"] != "d4_probability_mean":
        raise ValueError("Unexpected frozen checkpoint or inference policy")
    membership, fingerprints = development_rows(plan["checkpoint_sha256"])
    class_to_idx = {name: index for index, name in enumerate(predictor.classes)}
    known = development_pixels(membership, predictor.spec["input_size"], class_to_idx)
    sources = {}
    for name in plan["development_supported"]:
        raw, truth = known[name]
        signals, predicted = measurements(predictor, raw)
        sources[name] = (signals, predicted == truth)
        print(name, len(truth), int((predicted == truth).sum()), flush=True)
    negative = [dict(row, path=RAW / (row["image_id"] + ".jpg"))
                for row in rows(MANIFEST) if row["status"] == "included"]
    if len(negative) != 45 or any(row["split"] != "val" for row in negative) or len({r["sha256"] for r in negative}) != 45:
        raise ValueError("Unsupported cohort changed from registered development")
    signals, _ = measurements(predictor, pixels(negative, predictor.spec["input_size"]))
    groups = np.array([row["group"] for row in negative])
    unsupported = {}
    for group in plan["development_unsupported"]:
        mask = groups == group
        by_source = {name: {key: auroc(values[key], negative_values[mask])
                            for key, negative_values in signals.items()}
                     for name, (values, _) in sources.items()}
        unsupported[group] = {"count": int(mask.sum()), "by_source": by_source,
                              "source_mean": {key: float(np.mean([group[key] for group in by_source.values()]))
                                              for key in signals}}
    pooled_correct = np.concatenate([correct for _, correct in sources.values()])
    pooled_signals = {key: np.concatenate([values[key] for values, _ in sources.values()]) for key in signals}
    pooled = correctness_auc(pooled_signals, pooled_correct)
    guards = plan["advance"]
    passed = (all(item["source_mean"][CANDIDATE] >= guards["minimum_auc_in_each_unsupported_group"]
                  and item["source_mean"][CANDIDATE] - item["source_mean"][REFERENCE] >= guards["minimum_auc_gain_in_each_unsupported_group"]
                  for item in unsupported.values())
              and pooled[CANDIDATE] - pooled[REFERENCE] >= guards["minimum_pooled_correctness_auc_gain"])
    output.mkdir(parents=True)
    atomic_json(output / "result.json", {
        "created_utc": utc_now(), "checkpoint_sha256": predictor.metadata["checkpoint_sha256"], "plan_sha256": sha256(PLAN),
        "supported_fingerprints": fingerprints, "unsupported_manifest_sha256": sha256(MANIFEST),
        "unsupported": unsupported, "pooled_correctness": pooled,
        "by_source_correctness": {name: {"count": len(truth), "correct": int(truth.sum()), "auc": correctness_auc(scores, truth)}
                                  for name, (scores, truth) in sources.items()},
        "advance": passed, "test_opened": False, "threshold_fit": False, "deployment_changed": False,
    })
    print("pooled correctness", pooled, "advance", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
