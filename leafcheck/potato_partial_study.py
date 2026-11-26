"""Test a source of field photos with coarse potato blight labels."""

import argparse
from collections import Counter
import fcntl
from pathlib import Path
import signal

import numpy as np
import torch
from torch import nn

from .common import ROOT, atomic_json, environment, sha256, utc_now
from .data import cache_path
from .field_expansion import load_inputs
from .inference import Predictor
from .potato_study import FIELD_CONFIG, V2, run_arm, score_pixels
from .preprocessing import prepare_batch
from .train import device_for
from .upload_study import ARTIFACT, load_study, pixels, read, rows


PLAN = ROOT / "configs/leafcheck_potato_partial_v1.json"
RAW = ROOT / "data/raw/leafcheck_field_v9/iit-potato"
REGISTRY = ROOT / "data/manifests/leafcheck_iit_potato_intake_files.csv"


def partial_loss(logits, targets, weights):
    # Targets 0, 1 and 2 identify exact conditions. Target 3 allows either blight.
    logp = logits.log_softmax(1)
    exact = logp.gather(1, targets.clamp_max(2).unsqueeze(1)).flatten()
    grouped = torch.logsumexp(logp[:, (0, 2)], dim=1)
    per_item = -torch.where(targets == 3, grouped, exact)
    sample_weights = weights[targets]
    return (per_item * sample_weights).sum() / sample_weights.sum()


def train_partial_epoch(model, optimizer, raw, labels, spec, device, config, epoch, seed, weights):
    model.train()
    order = np.random.default_rng(seed + 100003 * epoch).permutation(len(labels))
    total_loss = 0.0
    for start in range(0, len(labels), config["batch_size"]):
        ix = order[start:start + config["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        x = prepare_batch(raw[ix], spec, device, "dihedral_color")
        loss = partial_loss(model(x), labels[ix].to(device), weights)
        if not loss.isfinite().item():
            raise FloatingPointError("Nonfinite partial-label loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False, error_if_nonfinite=True)
        optimizer.step()
        total_loss += loss.item() * len(ix)
    return total_loss / len(labels)


def source_rows(audit_directory, plan):
    audit_path = audit_directory / "audit.csv"
    receipt = read(audit_directory / "audit.json")
    if (receipt["audit_sha256"] != sha256(audit_path) or receipt["source_files_sha256"] != sha256(REGISTRY)
            or receipt["split_assigned"] or receipt["inference_run"]):
        raise ValueError("Expected the frozen intake trace with no evaluated predictions")
    records = [dict(r, path=RAW / (r["file_id"] + ".bin")) for r in rows(audit_path)
               if r["screen"] == "no_candidate"]
    counts = Counter(row["source_label"] for row in records)
    if set(counts) != {"healthy", "blight_unspecified"} or len({row["sha256"] for row in records}) != len(records):
        raise ValueError("The partial-label source has unexpected or repeated entries")
    if any(counts[label] < minimum for label, minimum in plan["minimum_source_views"].items()):
        raise ValueError(f"Insufficient screened source views: {counts}")
    return records, receipt


def train(output, audit_directory, resume):
    if (output / "decision.json").exists():
        print("Preserving the completed run", flush=True)
        return
    config_plan = read(PLAN)
    candidates, receipt = source_rows(audit_directory, config_plan)
    device = device_for("auto")
    torch.set_num_threads(4)
    config = read(ROOT / "configs" / config_plan["base_recipe"])
    config.update(min_epochs=config_plan["min_epochs"], max_epochs=config_plan["max_epochs"])
    seed = config_plan["seed"]
    classes = config["labels"]
    assert classes == ["potato__early_blight", "potato__healthy", "potato__late_blight"]
    lineage = read(FIELD_CONFIG)
    frozen, release, old, new = load_inputs(V2, lineage)
    if receipt["master_fingerprint"] != frozen["fingerprint"]:
        raise ValueError("The screened source belongs to a different master release")
    _, master, _ = load_study(read(ROOT / "configs/upload_blight_v1.json"))
    spec = read(cache_path(config["input_size"], frozen["fingerprint"]) / "preprocessing.json")
    run = {"protocol_sha256": sha256(PLAN), "master_fingerprint": frozen["fingerprint"],
           "field_fingerprint": release["fingerprint"], "iit_audit_sha256": receipt["audit_sha256"],
           "base_checkpoint": lineage["initial_weights"], "spec": spec, "config": config,
           "trainer_sha256": sha256(Path(__file__)),
           "checkpoint_loop_sha256": sha256(ROOT / "leafcheck/potato_study.py")}
    run_path = output / "run.json"
    if run_path.exists():
        if not resume or read(run_path)["run"] != run:
            raise ValueError("Use --resume with the unchanged registered run")
    else:
        atomic_json(run_path, {"run": run, "environment": environment(device), "started_utc": utc_now()})

    standard = [r for r in master + old + new if r["split"] == "train" and r["class_name"] in classes]
    development = {
        "master": [r for r in master if r["split"] == "val" and r["class_name"] in classes],
        "field": [r for r in new if r["split"] == "val" and r["source"] == "mduma" and r["class_name"] in classes],
    }
    validation_ids = {r["sha256"] for members in development.values() for r in members}
    if validation_ids.intersection(r["sha256"] for r in standard + candidates):
        raise ValueError("A training image appears in validation")

    def samples(records, size):
        return pixels(records, size), torch.tensor([classes.index(r["class_name"]) for r in records])

    original = Predictor(ARTIFACT, device)
    columns = [original.classes.index(name) for name in classes]
    baseline = {name: score_pixels(original.scoring_model, samples(records, original.spec["input_size"]),
                                   original.spec, classes, device, config["batch_size"], columns)
                for name, records in development.items()}
    val = {name: samples(records, config["input_size"]) for name, records in development.items()}
    train_raw, train_labels = samples(standard, config["input_size"])
    coarse = torch.tensor([1 if r["source_label"] == "healthy" else 3 for r in candidates])
    random = np.random.default_rng(seed)
    matching = {label: np.flatnonzero(train_labels.numpy() == label) for label in range(3)}
    blight_count = 0
    donor_indices = []
    for target in coarse.tolist():
        fine = 1 if target == 1 else (0, 2)[blight_count % 2]
        blight_count += target == 3
        donor_indices.append(int(random.choice(matching[fine])))
    study_targets = torch.cat((train_labels, coarse))
    extra = {"control": train_raw[donor_indices], "field": pixels(candidates, config["input_size"])}
    atomic_json(output / "membership.json", {
        "shared_training_ids": [r["image_id"] for r in standard],
        "donor_training_ids": [standard[i]["image_id"] for i in donor_indices],
        "iit_asset_ids": [r["file_id"] for r in candidates],
        "coarse_targets": coarse.tolist(),
        "validation": {key: [r["image_id"] for r in subset] for key, subset in development.items()},
    })
    stop = {"requested": False}

    def request_stop(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print("Training", len(standard), "+", len(candidates), "screened views per arm; validation",
          {k: len(v) for k, v in development.items()}, flush=True)
    results = {}
    for arm, supplemental in extra.items():
        result = run_arm(output / arm, config, arm, seed, (torch.cat((train_raw, supplemental)), study_targets),
                         val, baseline, spec, device, stop, run, train_epoch=train_partial_epoch)
        if result is None or stop["requested"]:
            return
        results[arm] = result
    before = results["control"]["scores"]
    after = results["field"]["scores"]
    field_gain = after["field"]["macro_f1"] - before["field"]["macro_f1"]
    class_recall_deltas = {a["class_name"]: a["recall"] - b["recall"]
                           for a, b in zip(after["field"]["per_class"], before["field"]["per_class"], strict=True)}
    master_deltas = {key: after["master"][key] - before["master"][key] for key in ("macro_f1", "accuracy")}
    passed = (results["field"]["passed_master_limits"]
              and field_gain >= config_plan["min_gain_field_macro_f1"]
              and min(class_recall_deltas.values()) >= -config_plan["max_drop_field_recall_per_label"]
              and master_deltas["macro_f1"] >= -config_plan["max_drop_master_macro_f1"]
              and master_deltas["accuracy"] >= -config_plan["max_drop_master_accuracy"])
    atomic_json(output / "decision.json", {
        "run": run, "screened_source_counts": Counter(row["source_label"] for row in candidates),
        "membership_sha256": sha256(output / "membership.json"), "completed_utc": utc_now(),
        "arms": results, "masked_incumbent": baseline, "field_macro_f1_gain": field_gain,
        "field_recall_deltas": class_recall_deltas, "master_deltas": master_deltas,
        "passed_screen": passed, "test_used": False, "deployment_changed": False,
    })
    print("Field gain", field_gain, "Passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".train.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args.output, args.audit, args.resume)
