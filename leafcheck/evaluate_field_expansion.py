"""Evaluate the frozen field candidate and reference on the new sources once."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import shutil

import numpy as np
import torch

from .common import atomic_json, sha256, utc_now, write_csv
from .data import split_rows
from .inference import Predictor
from .preprocessing import prepare_batch
from .route_field import routed
from .train import device_for
from .upload_study import ARTIFACT, pixels, read, rows, score, summarize


def freeze(output, decision_path, release_path, routing_path=None):
    decision = read(decision_path)
    release = read(release_path / "release.json")
    candidate = decision["arms"]["new_field"]
    if release["fingerprint"] != decision["release"]:
        raise ValueError("This decision and release do not belong to the same study")
    checkpoint = Path(candidate["checkpoint"])
    if sha256(checkpoint) != candidate["sha256"]:
        raise ValueError("The selected experimental checkpoint changed")
    lock = output / "freeze.json"
    if lock.exists():
        raise FileExistsError("Preserving the existing test freeze")
    record = {
        "frozen_utc": utc_now(), "decision_sha256": sha256(decision_path),
        "data_manifest_sha256": release["files_sha256"]["manifest.csv"], "release": release["fingerprint"],
        "class_map_sha256": sha256(ARTIFACT / "class_to_idx.json"),
        "preprocessing_sha256": sha256(ARTIFACT / "preprocessing.json"),
        "reference_sha256": decision["base_checkpoint"], "candidate_sha256": candidate["sha256"],
        "candidate_checkpoint": str(checkpoint), "policy": "d4_probability_mean",
        "primary_metric": "Macro-F1 over each test source's supported labels. All 177 outputs eligible.",
        "deployment_guards_passed_on_validation": decision["promotion_guards_passed"],
        "rule": "Frozen master training class frequencies, restricted independently to each source's true label set",
    }
    if routing_path:
        routing = read(routing_path)
        if (not routing["promotion_guards_passed"] or routing["release"] != release["fingerprint"]
                or routing["parent_decision_sha256"] != record["decision_sha256"]):
            raise ValueError("Routing decision is ineligible or references other weights")
        record["experimental_routing"] = {
            "selection_sha256": sha256(routing_path),
            "reference_below": routing["plan"]["baseline_score_threshold"],
            "candidate_at_least": routing["selected_options"]["new_field"]["candidate_min"],
        }
        record["deployment_guards_passed_on_validation"] = True
    atomic_json(lock, record)


@torch.inference_mode()
def run_model(predictor, raw):
    return torch.cat([
        predictor.scoring_model(prepare_batch(batch, predictor.spec, predictor.device)).softmax(1).cpu()
        for batch in raw.split(96)
    ]).numpy()


def evaluate(output, release):
    if (output / "metrics.json").exists():
        raise FileExistsError("Test evaluation already recorded")
    lock = read(output / "freeze.json")
    if sha256(release / "manifest.csv") != lock["data_manifest_sha256"]:
        raise ValueError("Test membership differs from the freeze")
    for name, key in (("class_to_idx.json", "class_map_sha256"), ("preprocessing.json", "preprocessing_sha256")):
        if sha256(ARTIFACT / name) != lock[key]:
            raise ValueError(f"Reference {name} differs from the freeze")
    device = device_for("auto")
    torch.set_num_threads(4)
    incumbent = Predictor(ARTIFACT, device)
    if incumbent.metadata["checkpoint_sha256"] != lock["reference_sha256"] or incumbent.policy["name"] != lock["policy"]:
        raise ValueError("The reference checkpoint or prediction policy changed")
    member = Path(lock["candidate_checkpoint"])
    if sha256(member) != lock["candidate_sha256"]:
        raise ValueError("Experimental weights differ from the freeze")
    records = [dict(r, path=release / r["output_relative_path"])
               for r in rows(release / "manifest.csv") if r["status"] == "included" and r["split"] == "test"]
    if not records or len(records) != len({r["image_id"] for r in records}):
        raise ValueError("The test set is empty or has repeated IDs")
    assert all(incumbent.classes[int(r["class_id"])] == r["class_name"] for r in records)
    raw = pixels(records, incumbent.spec["input_size"])
    truth = np.array([int(r["class_id"]) for r in records], dtype=np.int64)
    origins = np.array([r["source"] for r in records])
    baseline = run_model(incumbent, raw)
    state = torch.load(member, map_location="cpu", weights_only=True)
    incumbent.model.load_state_dict(state["weights"], strict=True)
    fresh = run_model(incumbent, raw)
    route = lock.get("experimental_routing")
    switches = 0
    if route:
        fresh, switches = routed(baseline, fresh, route["reference_below"], route["candidate_at_least"])
    classes = incumbent.classes
    frequencies = Counter(r["class_name"] for r in split_rows("train"))
    measures, predictions = {}, []
    for source in sorted(set(origins)):
        take = np.flatnonzero(origins == source)
        supported = np.unique(truth[take])
        rule = max(supported.tolist(), key=lambda label: (frequencies[classes[label]], -label))
        measures[source] = {}
        options = {"reference": baseline[take].argmax(1), "candidate": fresh[take].argmax(1),
                   "training_frequency_rule": np.full(len(take), rule, dtype=np.int64)}
        old, new = options["reference"] == truth[take], options["candidate"] == truth[take]
        measures[source]["paired_correctness"] = {
            "both_correct": int((old & new).sum()), "both_wrong": int((~old & ~new).sum()),
            "candidate_only_correct": int((~old & new).sum()), "reference_only_correct": int((old & ~new).sum())}
        for method, labels in options.items():
            full = score(truth[take], labels, classes, supported)
            measures[source][method] = dict(summarize(full), blight=[
                entry for entry in full["blight"] if entry["total"]
            ], per_class=[full["per_class"][int(i)] for i in supported], largest_errors=full["largest_errors"])
        for i in take:
            before, after = int(baseline[i].argmax()), int(fresh[i].argmax())
            predictions.append({
                "image_id": records[i]["image_id"], "source": source, "class_name": classes[truth[i]],
                "reference_prediction": classes[before], "reference_score": float(baseline[i, before]),
                "candidate_prediction": classes[after], "candidate_score": float(fresh[i, after]),
                "rule_prediction": classes[rule], "path": records[i]["output_relative_path"],
            })
    write_csv(output / "predictions.csv", predictions)
    atomic_json(output / "metrics.json", {"freeze_sha256": sha256(output / "freeze.json"),
                                         "predictions_sha256": sha256(output / "predictions.csv"),
                                         "evaluated_utc": utc_now(), "groups": measures,
                                         "router_switches": switches})
    if route:
        staged = output / "model.pt.tmp"
        shutil.copyfile(member, staged)
        if sha256(staged) != lock["candidate_sha256"]:
            raise ValueError("Experimental checkpoint changed during bundling")
        staged.replace(output / "model.pt")
        atomic_json(output / "bundle.json", {
            "version": "leafcheck177-field-study-v2",
            "files_sha256": {name: sha256(output / name) for name in ("model.pt", "freeze.json", "metrics.json")},
        })
    print({source: {name: {k: value[k] for k in ("correct", "sample_count", "macro_f1", "wrong_crop")}
                    for name, value in data.items() if name != "paired_correctness"}
           for source, data in measures.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--routing", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("This evaluation already has an owner") from error
        if args.action == "freeze":
            if not args.decision:
                parser.error("--decision is required for freeze")
            freeze(args.output, args.decision, args.release, args.routing)
        else:
            evaluate(args.output, args.release)


if __name__ == "__main__":
    main()
