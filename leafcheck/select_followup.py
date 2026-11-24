"""Compare probability blends and uncertainty rules on development data only."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from .common import ROOT, atomic_json, sha256
from .refit_study import split_scores
from .reliability_study import measures
from .upload_study import read, rows, summarize

PROTOCOL = ROOT / "configs/leafcheck_followup_v1.json"


def signal_from_probs(p, rule):
    if rule == "max_softmax":
        return p.max(1)
    if rule == "top_two_margin":
        top = np.partition(p, -2, axis=1)[:, -2:]
        return top[:, 1] - top[:, 0]
    raise ValueError(f"Unsupported uncertainty rule: {rule}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    root = args.run
    config = read(PROTOCOL)
    classes = list(read(ROOT / "data/manifests/leafcheck177_frozen.json")["classes"])
    development = read(root / "reliability_development.json")
    master_p = np.load(root / "incumbent_validation_probabilities.npy")
    head = root / config["head_recipe"]
    adapted_p = np.load(head / "validation_probabilities.npy")
    assert read(head / "status.json")["state"] == "complete"
    members = rows(root / "val.csv")
    baseline = split_scores(master_p, members, classes)
    blends = []
    for mix in config["head_mixture_weights"]:
        metric = split_scores((1 - mix) * master_p + mix * adapted_p, members, classes)
        m, s = summarize(metric["master"]), summarize(metric["supplement"])
        prereg = config["head_promotion"]
        eligible = (m["macro_f1"] - baseline["master"]["macro_f1"] >= prereg["minimum_master_macro_f1_gain"]
                    and baseline["master"]["accuracy"] - m["accuracy"] <= prereg["maximum_master_accuracy_decline"]
                    and s["macro_f1"] - baseline["supplement"]["macro_f1"] >= prereg["minimum_supplement_macro_f1_gain"])
        blends.append({"adapted_weight": mix, "master": m, "supplement": s, "eligible": eligible})
    # This study registers a screen for the incumbent only. A passing new
    # classifier would require a matching separate freeze and screen validation.
    assert not any(row["eligible"] for row in blends), "A new classifier qualifies; review its freeze"
    truth = np.array([int(r["class_id"]) for r in members])
    correct = master_p.argmax(1) == truth
    train = np.array([r["master"] == "True" for r in members])
    belong = np.array([int(hashlib.sha256(("reliability-v1:" + r["group_id"]).encode()).hexdigest()[:8], 16)
                       % 100 < 40 for r in members])
    calibration = train & belong
    check = train & ~belong
    supp = ~train & np.array([r["source"] != "bugwood" for r in members])
    baseline_check = measures(correct[check], np.ones(correct[check].shape, dtype=bool))
    candidates = {}
    for name in config["screen_rule_candidates"]:
        values = signal_from_probs(master_p, name)
        thresh = float(np.quantile(values[calibration], .10))
        accepted = values >= thresh
        on_check = measures(correct[check], accepted[check])
        on_supplement = measures(correct[supp], accepted[supp])
        qualifies = (on_check["coverage"] >= config["screen_minimum_master_check_coverage"]
                     and on_check["accepted_accuracy"] - baseline_check["accepted_accuracy"] >=
                     config["screen_minimum_master_check_accuracy_gain"]
                     and on_supplement["coverage"] <= config["screen_maximum_supplement_coverage"])
        candidates[name] = {"threshold": thresh, "check_master": on_check,
                            "supplement": on_supplement, "qualifies": qualifies}
    eligible = [name for name, value in candidates.items() if value["qualifies"]]
    best = min(eligible, key=lambda name: (candidates[name]["check_master"]["incorrect_per_upload"],
                                          candidates[name]["supplement"]["coverage"])) if eligible else None
    result = {"protocol_sha256": sha256(PROTOCOL), "checkpoint_sha256": development["base_checkpoint"],
              "class_map_sha256": sha256(ROOT / "artifacts/leafcheck177-v1/class_to_idx.json"),
              "features_sha256": sha256(root / "features.json"),
              "tested_head_sha256": sha256(head / "head.pt"),
              "blends": blends, "screen": best, "baseline_check": baseline_check, "screen_candidates": candidates}
    atomic_json(root / "followup_decision.json", result)
    for record in blends:
        print("BLEND", record, flush=True)
    for name, record in candidates.items():
        print("SCREEN", name, record, flush=True)
    print("SELECTED", best, flush=True)


if __name__ == "__main__":
    main()
