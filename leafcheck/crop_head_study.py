"""Test whether a scratch crop head helps the frozen 177-label model on field validation."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import signal

import numpy as np
import torch

from .common import ROOT, atomic_json, environment, metrics_from_matrix, sha256, utc_now
from .data import PixelCache, cache_path, release as master_release
from .field_expansion import load_inputs
from .inference import Predictor
from .potato_policy import distribution, master_probabilities
from .potato_study import run_arm
from .tomato_study import evaluate
from .train import device_for
from .upload_study import ARTIFACT, pixels, read, rows, summarize


PLAN = ROOT / "configs/leafcheck_crop_head_v1.json"
DATA = ROOT / "data/processed"
TRAIN = ("leafcheck_field_v2_sampled", "leafcheck_tom2024_v1")
DEV = "leafcheck_pakistan_tomato_dev_v1"
CCMT = "leafcheck_ccmt_crop_v1"
SAFARI = "leafcheck_imagesafari_crop_v1"


def checked_safari(plan):
    path = DATA / SAFARI
    release = read(path / "release.json")
    pinned = read(ROOT / f"data/manifests/{SAFARI}_release.json")
    hashes = {name: sha256(path / name) for name in pinned["files_sha256"]}
    if (release != pinned or hashes != pinned["files_sha256"]
            or release["fingerprint"] != plan["imagesafari_fingerprint"]):
        raise ValueError("Image Safari pilot requires the pinned crop-only release")
    records = [dict(r, path=path / r["output_relative_path"]) for r in rows(path / "manifest.csv")
               if r["status"] == "included"]
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    if not train or not val or ({r["group_id"] for r in train} & {r["group_id"] for r in val}):
        raise ValueError("Image Safari training and validation must use disjoint sites")
    return train, val


def checked_ccmt(frozen):
    path = DATA / CCMT
    release = read(path / "release.json")
    if release != read(ROOT / f"data/manifests/{CCMT}_release.json"):
        raise ValueError("The CCMT release differs from its registered copy")
    hashes = {name: sha256(path / name) for name in release["files_sha256"]}
    if (release["base_release"] != frozen["fingerprint"] or hashes != release["files_sha256"]
            or hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest() != release["fingerprint"]):
        raise ValueError("The CCMT crop release failed its integrity check")
    return release["fingerprint"], [
        dict(row, path=path / row["output_relative_path"]) for row in rows(path / "manifest.csv")
        if row["status"] == "included" and row["split"] == "train"
    ]


def load_sources(plan, frozen):
    lineage = {"initial_weights": plan["reference_checkpoint"]}
    releases = {}
    previous = None
    contents = {}
    for name in (*TRAIN, DEV):
        attached, release, base, members = load_inputs(DATA / name, lineage)
        if attached["fingerprint"] != frozen["fingerprint"]:
            raise ValueError("Unexpected base release")
        previous = base
        releases[name] = release["fingerprint"]
        contents[name] = members
    releases[CCMT], photos = checked_ccmt(frozen)
    # PlantSeg regions are derived crops. The initial screen asks about original field views.
    extra = [r for r in previous + contents[TRAIN[0]] + contents[TRAIN[1]] if r["split"] == "train"]
    development = {name: [r for r in previous + contents[TRAIN[0]]
                          if r["source"] == name and r["split"] == "val"]
                   for name in plan["known_development"] if name != "pakistan"}
    development["pakistan"] = contents[DEV]
    extra = [dict(r, crop=r["class_name"].split("__", 1)[0]) for r in extra] + photos
    dev = [r for samples in development.values() for r in samples]
    if (not all(development.values()) or any(r["split"] != "val" for r in dev)
            or {r["group_id"] for r in extra} & {r["group_id"] for r in dev}
            or len({r["pixel_d4_sha256"] for r in extra + dev}) != len(extra) + len(dev)):
        raise ValueError("Extra training and development sets must be separate")
    return releases, extra, development


def blend_crop(probabilities, crop_probabilities, crop_index, mix):
    membership = np.eye(crop_probabilities.shape[1], dtype=probabilities.dtype)[crop_index]
    original_crop = probabilities @ membership
    # Replace part of the crop mass while preserving condition ranking inside each crop.
    new_mass = (1 - mix) * original_crop + mix * crop_probabilities
    factors = new_mass / np.maximum(original_crop, np.finfo(probabilities.dtype).tiny)
    adjusted = probabilities * factors[:, crop_index]
    return adjusted / adjusted.sum(1, keepdims=True)


def measures(probabilities, y, classes):
    result = evaluate(probabilities, y, classes)
    return dict(summarize(result), loss=result["loss"], per_class=result["per_class"],
                blight=[row for row in result["blight"] if row["total"]],
                largest_errors=result["largest_errors"], confusion_matrix=[
                    [i, j, count] for i, row in enumerate(result["confusion_matrix"])
                    for j, count in enumerate(row) if count
                ])


def run(output, plan, seed, resume):
    report = output / "decision.json"
    if report.exists():
        raise FileExistsError(report)
    torch.set_num_threads(4)
    device = device_for("auto")
    incumbent = Predictor(ARTIFACT, device)
    frozen = master_release()
    classes = incumbent.classes
    if (frozen["fingerprint"] != plan["master_fingerprint"]
            or incumbent.metadata["checkpoint_sha256"] != plan["reference_checkpoint"]):
        raise ValueError("Reference identity changed")
    releases, supplemental, development = load_sources(plan, frozen)
    crop_names, crop_index = np.unique([name.split("__", 1)[0] for name in classes], return_inverse=True)
    crop_to_index = {name: i for i, name in enumerate(crop_names)}
    config = {**plan, "labels": crop_names.tolist()}
    train = PixelCache("train", plan["input_size"], frozen)
    validation = PixelCache("val", plan["input_size"], frozen)
    spec = read(cache_path(plan["input_size"], frozen["fingerprint"]) / "preprocessing.json")
    sampler = np.random.default_rng(seed)
    pools = [np.flatnonzero(crop_index[train.labels] == i) for i in range(len(crop_names))]
    selected = [sampler.permutation(pool)[:plan["master_per_crop_cap"]] for pool in pools]
    base = np.concatenate(selected)
    safari = "imagesafari_fingerprint" in plan
    shared = supplemental if safari else []
    if safari:
        safari_train, safari_val = checked_safari(plan)
        repeats = plan["imagesafari_train_occurrences_per_original"]
        candidate_extra = safari_train * repeats
        releases[SAFARI] = plan["imagesafari_fingerprint"]
    else:
        candidate_extra = supplemental
        safari_val = []
    new_crop_indices = np.array([crop_to_index[row["crop"]] for row in candidate_extra])
    replay = np.concatenate([sampler.choice(pool, int((new_crop_indices == i).sum()), replace=True)
                             for i, pool in enumerate(selected)])
    train_groups = {train.rows[i]["group_id"] for i in base} | {row["group_id"] for row in shared + candidate_extra}
    val_groups = {row["group_id"] for row in validation.rows}
    val_groups.update(row["group_id"] for samples in development.values() for row in samples)
    val_groups.update(row["group_id"] for row in safari_val)
    if train_groups & val_groups:
        raise ValueError("A validation group appears in the crop training set")
    if not np.array_equal(np.sort(crop_index[train.labels[replay]]), np.sort(new_crop_indices)):
        raise ValueError("Control and treatment crop occurrences differ")
    run_record = {
        "config": config, "seed": seed, "releases": releases, "reference_spec": incumbent.spec,
        "crop_spec": spec, "training_order_sha256": hashlib.sha256(np.concatenate([base, replay]).tobytes()).hexdigest(),
        "trainer": sha256(Path(__file__)), "helper": sha256(Path(__file__).with_name("potato_study.py")),
        "training": {"master": len(base), "shared_field": len(shared), "field": len(candidate_extra),
                     "unique_new_field": len({r["image_id"] for r in candidate_extra}),
                     "by_field_crop": dict(Counter(row["crop"] for row in candidate_extra))},
    }
    receipt = output / "run.json"
    if receipt.exists():
        if not resume or read(receipt)["run"] != run_record:
            raise ValueError("Resume requires the matching training recipe and inputs")
    else:
        atomic_json(receipt, {"run": run_record, "environment": environment(device), "started_utc": utc_now()})

    original_val = PixelCache("val", incumbent.spec["input_size"], frozen)
    if [r["image_id"] for r in original_val.rows] != [r["image_id"] for r in validation.rows]:
        raise ValueError("Validation caches disagree on image membership or order")
    inputs = {"master": validation.batch(slice(None))[0]}
    labels = {"master": validation.labels}
    original = {"master": master_probabilities(incumbent, original_val)}
    for name, records in development.items():
        inputs[name] = pixels(records, plan["input_size"])
        labels[name] = np.array([int(r["class_id"]) for r in records])
        original[name] = distribution(incumbent.scoring_model, pixels(records, incumbent.spec["input_size"]),
                                      incumbent.spec, device, size=plan["batch_size"])
    ref = {name: measures(probability, labels[name], classes) for name, probability in original.items()}
    field_sources = plan["known_development"]
    print("Original field:", {name: (ref[name]["correct"], ref[name]["wrong_crop"]) for name in field_sources}, flush=True)

    def pack(by_source):
        field = {key: float(np.mean([by_source[name][key] for name in field_sources]))
                 for key in ("macro_f1", "accuracy", "loss")}
        field["confusion_matrix"] = {name: by_source[name]["confusion_matrix"] for name in field_sources}
        return {"master": by_source["master"], "field": field, "by_source": by_source}

    baseline = pack(ref)
    if safari:
        safari_truth = np.asarray([crop_to_index[row["crop"]] for row in safari_val])
        safari_raw = pixels(safari_val, plan["input_size"])
        raw_original = pixels(safari_val, incumbent.spec["input_size"])
        safari_original = distribution(incumbent.scoring_model, raw_original, incumbent.spec, device)

        def safari_score(joint):
            prediction = crop_index[joint.argmax(1)]
            n = len(crop_names)
            matrix = np.bincount(safari_truth * n + prediction, minlength=n * n).reshape(n, n)
            result = metrics_from_matrix(matrix, config["labels"])
            supported = [row for row in result["per_class"] if row["support"]]
            return {"macro_f1": float(np.mean([row["f1"] for row in supported])),
                    "accuracy": result["accuracy"], "correct": result["correct"],
                    "sample_count": result["sample_count"], "per_crop": supported}

        baseline["site_held_crop"] = safari_score(safari_original)

    def assess(scorer):
        crop_network = scorer.network
        scores = {}
        for name, raw in inputs.items():
            crop_prob = distribution(crop_network, raw, spec, device, size=plan["batch_size"])
            mixed = blend_crop(original[name], crop_prob, crop_index, plan["crop_blend"])
            scores[name] = measures(mixed, labels[name], classes)
            scores[name]["crop_head_accuracy"] = float(np.mean(crop_prob.argmax(1) == crop_index[labels[name]]))
        combined = pack(scores)
        if safari:
            crop_prob = distribution(crop_network, safari_raw, spec, device, size=plan["batch_size"])
            combined["site_held_crop"] = safari_score(blend_crop(safari_original, crop_prob, crop_index, plan["crop_blend"]))
        return combined

    stop = {"requested": False}

    def request_stop(*_):
        stop["requested"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    base_raw, base_ids = train.batch(base)
    control_raw, control_ids = train.batch(replay)
    base_crop = crop_index[base_ids]
    if safari:
        base_raw = torch.cat([base_raw, pixels(shared, plan["input_size"])])
        base_crop = np.concatenate([base_crop, [crop_to_index[r["crop"]] for r in shared]])
        field_pixels = pixels(safari_train, plan["input_size"]).repeat((repeats, 1, 1, 1))
    else:
        field_pixels = pixels(candidate_extra, plan["input_size"])
    arms = {
        "control": (torch.cat([base_raw, control_raw]),
                    torch.from_numpy(np.concatenate([base_crop, crop_index[control_ids]]))),
        "field": (torch.cat([base_raw, field_pixels]),
                  torch.from_numpy(np.concatenate([base_crop, new_crop_indices]))),
    }
    result = {}
    for name, samples in arms.items():
        result[name] = run_arm(output / name, config, name, seed, samples, None, baseline, spec, device,
                               stop, run_record, assess)
        if result[name] is None:
            return
    treated = result["field"]["scores"]
    ref_f1 = baseline["field"]["macro_f1"]
    control_f1 = result["control"]["scores"]["field"]["macro_f1"]
    field_f1 = treated["field"]["macro_f1"]
    per_source = {name: treated["by_source"][name]["macro_f1"] - ref[name]["macro_f1"] for name in field_sources}
    crop_error_reduction = float(np.mean([
        (ref[name]["wrong_crop"] - treated["by_source"][name]["wrong_crop"]) / ref[name]["sample_count"]
        for name in field_sources
    ]))
    rules = plan["selection"]
    passed = (result["field"]["passed_master_limits"] and field_f1 - ref_f1 >= rules["field_min_macro_f1_gain"]
              and field_f1 - control_f1 >= rules["field_min_gain_vs_control"]
              and min(per_source.values()) >= -rules["field_max_per_source_drop"]
              and crop_error_reduction >= rules["field_min_wrong_crop_rate_reduction"])
    site_decision = {}
    if safari:
        three = [x["site_held_crop"]["macro_f1"] for x in (baseline, result["control"]["scores"], treated)]
        site_decision = {"gain_over_reference": three[2] - three[0], "gain_over_control": three[2] - three[1]}
        passed = (result["field"]["passed_master_limits"]
                  and site_decision["gain_over_reference"] >= plan["min_imagesafari_crop_macro_f1_gain_vs_joint_reference"]
                  and site_decision["gain_over_control"] >= plan["min_imagesafari_crop_macro_f1_gain_vs_control"]
                  and field_f1 >= control_f1 - plan["max_joint_field_drop_vs_control"]
                  and min(per_source.values()) >= -rules["field_max_per_source_drop"])
    atomic_json(report, {"completed": utc_now(), "run": run_record, "baseline": baseline, "arms": result,
                         "gain_over_reference": field_f1 - ref_f1, "gain_over_control": field_f1 - control_f1,
                         "field_source_gains": per_source, "field_crop_error_reduction": crop_error_reduction,
                         "site_held_crop_decision": site_decision, "passed_screen": passed, "test_opened": False})
    print("Field macro-F1", field_f1, "control", control_f1, "reference", ref_f1, "passed", passed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(434, 435), default=434)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(args.plan)
        if plan.get("inherits") == PLAN.name:
            plan = {**read(PLAN), **plan}
        run(args.output, plan, args.seed, args.resume)
