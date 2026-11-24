"""Validation-only blight specialist experiment alongside the frozen 177-label CNN."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .common import (DATA, ROOT, atomic_checkpoint, atomic_json, environment,
                     metrics_from_matrix, seed_everything, sha256, update_matrix, write_csv)
from .data import cache_path, release
from .inference import Predictor
from .model import LeafCNN
from .preprocessing import image_bytes, prepare_batch
from .train import cpu_state, device_for, restore_rng, rng_state

ARTIFACT = ROOT / "artifacts/leafcheck177-v1"
SUPPLEMENT = DATA.parent / "leafcheck_blight_v1"
SAVED_VALIDATION = ROOT / ("runs/final177/inference_comparison/"
                           "residual177_224_seed434-d4_probability_mean-predictions.csv")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def source(row):
    if row.get("source"):
        return row["source"]
    if row["documented_leaf_ids"]:
        return "documented_plantvillage"
    if row["plantdoc_source_paths"]:
        return "documented_plantdoc_match"
    return "other_master_source_unresolved"


def member_rows(directory, mapping):
    members = []
    for row in rows(directory / "manifest.csv"):
        if row["status"] != "included" or row["split"] not in ("train", "val"):
            continue
        assert mapping[row["class_name"]] == int(row["class_id"])
        members.append({**row, "path": directory / row["output_relative_path"], "source_slice": source(row)})
    return members


def load_study(config):
    frozen = release()
    assert frozen["fingerprint"] == config["dataset_fingerprint"]
    assert frozen["files"] == read(ROOT / "data/manifests/leafcheck177_frozen.json")["files"]
    assert sha256(ARTIFACT / "model.pt") == config["base_checkpoint_sha256"]
    pin = read(ROOT / "data/manifests/leafcheck_blight_v1_release.json")
    assert pin["fingerprint"] == config["supplement_fingerprint"]
    assert pin["protected_release_fingerprint"] == frozen["fingerprint"]
    assert all(sha256(SUPPLEMENT / name) == h for name, h in pin["files_sha256"].items())
    assert (SUPPLEMENT / "class_to_idx.json").read_bytes() == (DATA / "class_to_idx.json").read_bytes()
    master = member_rows(DATA, frozen["class_to_idx"])
    added = member_rows(SUPPLEMENT, frozen["class_to_idx"])
    assert len(master) == 59008 + 16836 and len(added) == 1146 + 280
    assert len({r["image_id"] for r in master + added}) == len(master) + len(added)
    assert len({r["pixel_d4_sha256"] for r in master + added}) == len(master) + len(added)
    group_splits = {}
    for row in master + added:
        assert group_splits.setdefault(row["group_id"], row["split"]) == row["split"]
    return frozen, master, added


def inventory(output, master, added):
    counts = {}
    for name, records in (("master", master), ("supplement", added)):
        for row in records:
            key = name, row["split"], row["source_slice"], row["class_name"]
            record = counts.setdefault(key, [0, set(), 0, 0])
            record[0] += 1
            record[1].add(row["group_id"])
            record[2] += bool(row.get("documented_leaf_ids"))
            record[3] += row.get("offline_augmentation") == "True"
    write_csv(output / "support_by_source.csv", [
        dict(release=name, split=split, source=origin, crop=label.split("__")[0], label=label,
             images=n, recorded_groups=len(groups), documented_leaf_id_files=known,
             recorded_offline_augmented=augmented)
        for (name, split, origin, label), (n, groups, known, augmented) in sorted(counts.items())
    ])


def pixels(records, size):
    def decode(row):
        assert sha256(row["path"]) == row["sha256"], row["path"]
        raw = image_bytes(row["path"], size)
        return np.frombuffer(raw, np.uint8).reshape(size, size, 3)

    with ThreadPoolExecutor(6) as pool:
        return torch.from_numpy(np.stack(list(pool.map(decode, records))))


@torch.inference_mode()
def probabilities(model, raw, spec, device, batch_size=96):
    model.eval()
    return np.concatenate([
        model(prepare_batch(batch, spec, device)).softmax(1).cpu().numpy()
        for batch in raw.split(batch_size)
    ])


def score(y, prediction, classes, average_indices=None):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    update_matrix(matrix, y, prediction)
    result = metrics_from_matrix(matrix, classes)
    if average_indices is not None:
        result["macro_f1"] = float(np.mean([result["per_class"][i]["f1"] for i in average_indices]))
    result["largest_errors"] = [
        {"truth": classes[i], "prediction": classes[j], "count": count}
        for count, i, j in sorted(((int(matrix[i, j]), i, j) for i, j in zip(*matrix.nonzero())
                                   if i != j), reverse=True)[:15]
    ]
    parts = [name.split("__", 1) for name in classes]
    crop = np.array([p[0] for p in parts])
    result["wrong_crop"] = int(np.count_nonzero(crop[y] != crop[prediction]))
    result["right_crop_wrong_condition"] = int(np.count_nonzero((crop[y] == crop[prediction]) & (y != prediction)))
    result["blight"] = []
    for i, (plant, disease) in enumerate(parts):
        if disease not in ("early_blight", "late_blight"):
            continue
        opposite = plant + "__" + ("late_blight" if disease == "early_blight" else "early_blight")
        j = classes.index(opposite) if opposite in classes else None
        cross = crop != plant
        result["blight"].append({"true": classes[i], "opposite_label": opposite if j is not None else None,
                                 "opposite_count": int(matrix[i, j]) if j is not None else None,
                                 "total": int(matrix[i].sum()), "correct": int(matrix[i, i]),
                                 "wrong_crop": int(matrix[i, cross].sum()),
                                 "right_crop_wrong_condition": int(matrix[i, ~cross].sum() - matrix[i, i])})
    return result


def summarize(result):
    return {k: result[k] for k in ("macro_f1", "accuracy", "correct", "sample_count", "wrong_crop",
                                  "right_crop_wrong_condition")}


def training_set(master, added, config, run):
    permitted = set(config["labels"])
    base_rows = [r for r in master if r["split"] == "train" and r["class_name"] in permitted]
    return sorted(base_rows + ([r for r in added if r["split"] == "train"] if run["supplement"] else []),
                  key=lambda r: r["image_id"])


def baseline(output, master, added, focus, device):
    evaluator = Predictor(ARTIFACT, device)
    classes = evaluator.classes
    saved = {r["image_id"]: r for r in rows(SAVED_VALIDATION)}
    val = sorted((r for r in master if r["split"] == "val"), key=lambda r: r["image_id"])
    assert len(saved) == len(val) and {r["image_id"] for r in val} == set(saved)
    for row in val:
        item = saved[row["image_id"]]
        assert row["class_id"] == item["actual_index"] and row["class_name"] == item["actual_class"]
        assert row["group_id"] == item["group_id"]
    full_pred = np.array([int(saved[r["image_id"]]["predicted_index"]) for r in val])
    gate = np.flatnonzero(np.isin(full_pred, focus))
    supplement = sorted((r for r in added if r["split"] == "val"), key=lambda r: r["image_id"])
    targets = [val[i] for i in gate] + supplement
    p = probabilities(evaluator.scoring_model, pixels(targets, 224), evaluator.spec, device)
    # On a near tie, re-evaluate only the rows whose device differs before using
    # the archived incumbent prediction in the full-set comparison.
    differs = np.flatnonzero(p[:len(gate)].argmax(1) != full_pred[gate])
    if len(differs):
        cpu = Predictor(ARTIFACT)
        for i in differs:
            result = cpu.predict(targets[i]["path"])
            p[i] = [result["class_scores"][name] for name in classes]
    assert np.array_equal(p[:len(gate)].argmax(1), full_pred[gate])
    np.save(output / "incumbent_scored_rows.npy", p)
    write_csv(output / "incumbent_scored_rows.csv", [
        dict(image_id=r["image_id"], split=r["split"], source=r["source_slice"], true=r["class_name"],
             predicted=classes[int(prob.argmax())], score=float(prob.max()),
             image_path=str(r["path"].relative_to(ROOT)), group_id=r["group_id"])
        for r, prob in zip(targets, p)
    ])
    return val, supplement, gate, full_pred, p


def fuse(incumbent, specialist, gate, focus, mix):
    combined = incumbent.copy()
    within = combined[np.ix_(gate, focus)]
    combined[np.ix_(gate, focus)] = (1 - mix) * within + mix * within.sum(1, keepdims=True) * specialist[gate]
    return combined.argmax(1)


def assess(classes, val, supplement, gate, full_pred, p, focus, specialist=None, mix=.5):
    val_targets = np.array([int(r["class_id"]) for r in val])
    supp_targets = np.array([int(r["class_id"]) for r in supplement])
    supp_types = np.array([r["source_slice"] for r in supplement])
    val_types = np.array([r["source_slice"] for r in val])
    val_after = full_pred.copy()
    supp_p = p[len(gate):]
    supp_before = supp_p.argmax(1)
    supp_after = supp_before
    if specialist is not None:
        val_after[gate] = fuse(p[:len(gate)], specialist[:len(gate)], np.arange(len(gate)), focus, mix)
        supp_after = fuse(supp_p, specialist[len(gate):], np.flatnonzero(np.isin(supp_before, focus)), focus, mix)
    use = supp_types != "bugwood"
    results = {"master_validation": score(val_targets, val_after, classes),
               "supplement_validation": score(supp_targets[use], supp_after[use], classes, np.unique(supp_targets[use]))}
    results["by_source"] = {}
    for source_names, y, prediction in ((val_types, val_targets, val_after), (supp_types, supp_targets, supp_after)):
        for name in sorted(set(source_names)):
            select = source_names == name
            results["by_source"][name] = score(y[select], prediction[select], classes, np.unique(y[select]))
    return results, val_after, supp_after


def network(config, device):
    model = LeafCNN(architecture="plain_v1", channels=tuple(config["channels"]))
    model.classifier[-1] = nn.Linear(config["channels"][-1], len(config["labels"]))
    return model.to(device=device, memory_format=torch.channels_last)


def run_epoch(model, optimizer, train_pixels, labels, spec, device, config, epoch, seed, weights):
    model.train()
    order = np.random.default_rng(seed + 100003 * epoch).permutation(len(labels))
    loss_sum = 0.0
    for start in range(0, len(labels), config["batch_size"]):
        ix = order[start:start + config["batch_size"]]
        optimizer.zero_grad(set_to_none=True)
        x = prepare_batch(train_pixels[ix], spec, device, "dihedral_color")
        loss = nn.functional.cross_entropy(model(x), labels[ix].to(device), weight=weights)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.item() * len(ix)
    return loss_sum / len(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/upload_blight_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train", choices=["original_unweighted", "expanded_unweighted", "expanded_weighted"])
    parser.add_argument("--seed", type=int, default=434)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output
    config = read(args.config)
    output.mkdir(parents=True, exist_ok=True)
    frozen, master, supplement_members = load_study(config)
    classes = frozen["classes"]
    focus = np.array([classes.index(name) for name in config["labels"]])
    inventory(output, master, supplement_members)
    device = device_for(args.device)
    torch.set_num_threads(4)
    val, supplement, gate, full_pred, p = baseline(output, master, supplement_members, focus, device)
    base, _, _ = assess(classes, val, supplement, gate, full_pred, p, focus)
    reference = read(ARTIFACT / "freeze.json")["deployment_validation"]
    assert abs(base["master_validation"]["macro_f1"] - reference["macro_f1"]) < 1e-12
    assert base["master_validation"]["correct"] == reference["correct"]
    atomic_json(output / "baseline.json", base)
    print("INCUMBENT", json.dumps({k: summarize(base[k]) for k in ("master_validation", "supplement_validation")}), flush=True)
    if not args.train:
        return
    run = config["runs"][args.train]
    run_dir = output / f"{args.train}_seed{args.seed}"
    if not args.resume:
        run_dir.mkdir(exist_ok=False)
    elif read(run_dir / "status.json")["state"] == "complete":
        raise ValueError("This run is complete")
    spec = read(cache_path(config["input_size"], frozen["fingerprint"]) / "preprocessing.json")
    fit = training_set(master, supplement_members, config, run)
    fit_pixels = pixels(fit, config["input_size"])
    labels = torch.tensor([config["labels"].index(r["class_name"]) for r in fit])
    evaluation_pixels = pixels([val[i] for i in gate] + supplement, config["input_size"])
    seed_everything(args.seed)
    model = network(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=2)
    counts = torch.bincount(labels, minlength=len(focus)).float()
    weights = (counts.mean() / counts).pow(run["weight_power"]).to(device)
    identity = {"configuration_sha256": sha256(args.config), "dataset_fingerprint": frozen["fingerprint"],
                "supplement_fingerprint": config["supplement_fingerprint"], "device": str(device),
                "training_membership_sha256": hashlib.sha256("\n".join(r["image_id"] for r in fit).encode()).hexdigest(),
                "preprocessing": spec, "counts": dict(zip(config["labels"], [int(v) for v in counts])),
                "seed": args.seed, "run": args.train}
    history, best, best_weights = [], None, None
    if args.resume:
        state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        assert state["identity"] == identity
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        history, best, best_weights = state["history"], state["best"], state["best_weights"]
    else:
        write_csv(run_dir / "train_manifest.csv", [
            {"image_id": r["image_id"], "path": str(r["path"].relative_to(ROOT)), "class": r["class_name"],
             "group": r["group_id"], "source": r["source_slice"], "sha256": r["sha256"]} for r in fit])
        atomic_json(run_dir / "configuration.json", config)
        atomic_json(run_dir / "environment.json", environment(device))
    for epoch in range(len(history) + 1, config["max_epochs"] + 1):
        start = time.perf_counter()
        loss = run_epoch(model, optimizer, fit_pixels, labels, spec, device, config, epoch, args.seed, weights)
        specialist = probabilities(model, evaluation_pixels, spec, device)
        result, _, _ = assess(classes, val, supplement, gate, full_pred, p, focus, specialist, config["mix"])
        primary = result["master_validation"]["macro_f1"]
        extra = result["supplement_validation"]["macro_f1"]
        scheduler.step(primary)
        headline = {"epoch": epoch, "macro_f1": primary, "supplement_macro_f1": extra,
                    "accuracy": result["master_validation"]["accuracy"], "training_loss": loss,
                    "seconds": time.perf_counter() - start}
        # Master macro-F1 is primary. In a tie use supplement macro-F1, then the earlier epoch.
        if best is None or primary > best["macro_f1"] + 1e-6 or (
                abs(primary - best["macro_f1"]) <= 1e-6 and extra > best["supplement_macro_f1"] + 1e-6):
            best, best_weights = headline, cpu_state(model)
        history.append(headline)
        atomic_checkpoint(run_dir / "last.pt", {"identity": identity, "weights": cpu_state(model),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(),
                          "history": history, "best": best, "best_weights": best_weights})
        write_csv(run_dir / "history.csv", history)
        atomic_json(run_dir / "status.json", {"state": "running", "best": best, "completed_epochs": epoch})
        print(args.train, json.dumps(headline), flush=True)
        if epoch >= config["min_epochs"] and epoch - best["epoch"] >= config["patience"]:
            break
    checkpoint = run_dir / "best_model.pt"
    atomic_checkpoint(checkpoint, {"identity": identity, "weights": best_weights})
    restored = network(config, device)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["weights"])
    specialist = probabilities(restored, evaluation_pixels, spec, device)
    result, val_pred, supp_pred = assess(classes, val, supplement, gate, full_pred, p, focus, specialist, config["mix"])
    assert abs(result["master_validation"]["macro_f1"] - best["macro_f1"]) < 1e-12
    assert abs(result["supplement_validation"]["macro_f1"] - best["supplement_macro_f1"]) < 1e-12
    atomic_json(run_dir / "validation_metrics.json", result)
    write_csv(run_dir / "predictions.csv", [
        dict(image_id=r["image_id"], source=r["source_slice"], true=r["class_name"],
             before=classes[int(before)], after=classes[int(after)], group_id=r["group_id"])
        for r, before, after in zip(val + supplement, np.concatenate([full_pred, p[len(gate):].argmax(1)]),
                                    np.concatenate([val_pred, supp_pred]))
    ])
    atomic_json(run_dir / "status.json", {"state": "complete", "best": best, "completed_epochs": len(history),
                "checkpoint_sha256": sha256(checkpoint), "reload_verified": True, "identity": identity})
    print("COMPLETE", args.train, json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
