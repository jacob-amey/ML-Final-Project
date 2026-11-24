"""Matched field replay with an explicit frozen reference and validation-only selection."""

import argparse
from collections import Counter
import fcntl
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import ROOT, atomic_checkpoint, atomic_json, environment, seed_everything, sha256, utc_now
from .data import PixelCache
from .inference import Predictor
from .model import LeafCNN
from .preprocessing import prepare_batch
from .scoring import PredictionScores
from .train import cpu_state, device_for, evaluate, restore_rng, rng_state
from .upload_study import ARTIFACT, load_study, member_rows, pixels, read, score, summarize


def load_inputs(release_dir, config):
    old_study = read(ROOT / "configs/upload_blight_v1.json")
    frozen, _, previous = load_study(old_study)
    release = read(release_dir / "release.json")
    if not all(sha256(release_dir / path) == digest for path, digest in release["files_sha256"].items()):
        raise ValueError("The field release differs from its build record")
    digest = hashlib.sha256(json.dumps(release["files_sha256"], sort_keys=True).encode()).hexdigest()
    if digest != release["fingerprint"] or frozen["fingerprint"] != release["base_release"]:
        raise ValueError("The field release does not match the master dataset")
    new = member_rows(release_dir, frozen["class_to_idx"])
    previous_ids = {r["image_id"] for r in previous}
    if previous_ids.intersection(r["image_id"] for r in new):
        raise ValueError("New field release overlaps the previous addition")
    if sha256(ARTIFACT / "model.pt") != config["initial_weights"]:
        raise ValueError("Unexpected reference weights")
    return frozen, release, previous, new


def prepare_development(previous, new, size):
    dev = {}
    for source, members in (("sheikh", previous), ("banu", previous), ("mduma", new)):
        records = [r for r in members if r["source"] == source and r["split"] == "val"]
        if records:
            labels = np.array([int(row["class_id"]) for row in records])
            dev[source] = (pixels(records, size), labels)
    if "sheikh" not in dev:
        raise ValueError("Registered tomato validation data is missing")
    return dev


@torch.inference_mode()
def field_scores(scoring, dev, classes, spec, device, batch_size):
    scoring.eval()
    fields = {}
    for name, (raw, labels) in dev.items():
        prediction = torch.cat([
            scoring(prepare_batch(batch, spec, device)).argmax(1).cpu()
            for batch in raw.split(batch_size)
        ]).numpy()
        result = score(labels, prediction, classes, np.unique(labels))
        fields[name] = dict(summarize(result), blight=[
            entry for entry in result["blight"] if entry["total"]
        ], largest_errors=result["largest_errors"], support={
            classes[int(i)]: int(np.count_nonzero(labels == i)) for i in np.unique(labels)
        })
    return fields


def validate(model, validation, dev, classes, spec, device, config, policy):
    scoring = PredictionScores(model, {"name": policy})
    main, _ = evaluate(scoring, validation, spec, device, classes, config["batch_size"])
    crops = np.array([name.split("__", 1)[0] for name in classes])
    main["wrong_crop"] = int(np.asarray(main["confusion_matrix"])[crops[:, None] != crops].sum())
    main["right_crop_wrong_condition"] = main["sample_count"] - main["correct"] - main["wrong_crop"]
    fields = field_scores(scoring, dev, classes, spec, device, config["batch_size"])
    return {
        "master": dict(summarize(main), loss=main["loss"]),
        "fields": fields,
        "selection_field_f1": float(np.mean([metrics["macro_f1"]
                                             for source, metrics in fields.items() if source != "banu"])),
    }


def choose(checkpoint, best):
    if best is None:
        return True
    delta = checkpoint["master"]["macro_f1"] - best["master"]["macro_f1"]
    return delta > 1e-6 or abs(delta) <= 1e-6 and checkpoint["master"]["loss"] < best["master"]["loss"] - 1e-6


def is_eligible(candidate, reference, rules):
    return (candidate["master"]["macro_f1"] >= reference["master"]["macro_f1"] - rules["master_max_drop"]
            and candidate["master"]["accuracy"] >= reference["master"]["accuracy"] - rules["accuracy_max_drop"]
            and candidate["selection_field_f1"] >= reference["selection_field_f1"] + rules["field_min_gain"]
            and all(result["macro_f1"] >= reference["fields"][name]["macro_f1"] - .02
                    for name, result in candidate["fields"].items() if name != "banu"))


def mixed_epoch(model, teacher, optimizer, cache, common, extra, extra_labels, spec, device, config, epoch, stop):
    generator = np.random.default_rng(config["seed"] + 100003 * epoch)
    amount = min(config["master_per_epoch"], len(cache))
    base_order = np.random.default_rng(config["seed"]).permutation(len(cache))
    base_order = np.roll(base_order, -(epoch - 1) * amount)[:amount]
    common_pixels, common_labels = common
    other_labels = np.concatenate((common_labels, extra_labels))
    if extra is None:
        lookup = {int(label): np.flatnonzero(cache.labels == label) for label in np.unique(extra_labels)}
        rows_from_master = [generator.choice(lookup[int(label)]) for label in extra_labels]
        extra = torch.from_numpy(np.array(cache.pixels[rows_from_master]))
    else:
        # Generate the same choices before the permutation in both arms.
        lookup_counts = Counter(cache.labels.tolist())
        for label in extra_labels:
            generator.choice(lookup_counts[int(label)])
    other_pixels = torch.cat((common_pixels, extra))
    other_pixels, other_labels = other_pixels.numpy(), other_labels
    order = generator.permutation(amount + len(other_labels))
    counts = np.bincount(cache.labels, minlength=177)
    weights = np.clip((counts.mean() / counts) ** .5, .2, 5.0)
    weights /= (weights * counts).sum() / len(cache)
    weights = torch.tensor(weights, device=device, dtype=torch.float32)

    model.train()
    if config["fixed_batchnorm_statistics"]:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
    seen, total = 0, 0.0
    for offset in range(0, len(order), config["batch_size"]):
        indexes = order[offset:offset + config["batch_size"]]
        base = indexes[indexes < amount]
        added = indexes[indexes >= amount] - amount
        images = np.concatenate((cache.pixels[base_order[base]], other_pixels[added]))
        labels = np.concatenate((cache.labels[base_order[base]], other_labels[added]))
        x = prepare_batch(torch.from_numpy(images), spec, device, config["augmentation"])
        ground = torch.from_numpy(labels).to(device)
        optimizer.zero_grad(set_to_none=True)
        logit = model(x)
        per_image = F.cross_entropy(logit, ground, reduction="none")
        weighted = per_image.clone()
        if len(base):
            t = config["distillation_temperature"]
            with torch.no_grad():
                reference = teacher(x[:len(base)]).div(t).softmax(1)
            distill = F.kl_div(logit[:len(base)].div(t).log_softmax(1), reference, reduction="none").sum(1) * t * t
            proportion = config["distillation_mix"]
            weighted[:len(base)] = (1 - proportion) * per_image[:len(base)] * weights[ground[:len(base)]] + proportion * distill
        loss = weighted.mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(indexes)
        seen += len(indexes)
        if stop["requested"]:
            # Replaying a partial epoch from its preceding checkpoint preserves the schedule and RNG.
            return None
    return total / seen


def run(release_dir, output, config, resume):
    torch.set_num_threads(4)
    device = device_for("auto")
    frozen, release, previous, new = load_inputs(release_dir, config)
    receipt = {"config": config, "release": release["fingerprint"],
               "master": frozen["fingerprint"], "device": environment(device),
               "trainer_sha256": sha256(Path(__file__)),
               "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    receipt_path = output / "run.json"
    if receipt_path.exists():
        old = read(receipt_path)
        if not resume or any(old[key] != receipt[key] for key in ("config", "release", "trainer_sha256")):
            raise ValueError("Use --resume with the original configuration and release")
    else:
        atomic_json(receipt_path, receipt)
    incumbent = Predictor(ARTIFACT, device)
    spec, classes = incumbent.spec, incumbent.classes
    train = PixelCache("train", spec["input_size"], frozen)
    val = PixelCache("val", spec["input_size"], frozen)
    dev = prepare_development(previous, new, spec["input_size"])
    old_train = [r for r in previous if r["split"] == "train"]
    new_train = [r for r in new if r["split"] == "train"]
    assert set(r["source"] for r in new_train).issubset({"bapari", "mduma"})
    assert new_train and len(train) == 59008 and len(val) == 16836
    common = pixels(old_train, spec["input_size"]), np.array([int(r["class_id"]) for r in old_train])
    extra = pixels(new_train, spec["input_size"])
    extra_labels = np.array([int(r["class_id"]) for r in new_train])
    stop = {"requested": False}

    def pause(signum, frame):
        stop["requested"] = True

    signal.signal(signal.SIGTERM, pause)
    signal.signal(signal.SIGINT, pause)
    reference_path = output / "baseline.json"
    if reference_path.exists():
        reference = read(reference_path)
    else:
        reference = {}
        for policy in ("single_view", "d4_probability_mean"):
            atomic_json(output / "status.json", {"state": "baseline", "policy": policy, "utc": utc_now()})
            reference[policy] = validate(incumbent.model, val, dev, classes, spec, device, config, policy)
            print("Baseline", policy, reference[policy]["master"], reference[policy]["selection_field_f1"], flush=True)
        atomic_json(reference_path, reference)
    if stop["requested"]:
        atomic_json(output / "status.json", {"state": "paused", "utc": utc_now()})
        return
    architecture = read(ARTIFACT / "config.json")["model"]
    rules = config["selection"]
    selected = {}
    for arm in ("matched_control", "new_field"):
        directory = output / arm
        directory.mkdir(exist_ok=True)
        seed_everything(config["seed"])
        model = LeafCNN(**architecture).to(device=device, memory_format=torch.channels_last)
        model.load_state_dict(incumbent.model.state_dict(), strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        latest = directory / "last.pt"
        state = {"weights": cpu_state(model), "optimizer": optimizer.state_dict(), "rng": rng_state(),
                 "history": [], "best": None, "eligible": None}
        if latest.exists():
            state = torch.load(latest, weights_only=True, map_location="cpu")
            model.load_state_dict(state["weights"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
        else:
            atomic_checkpoint(latest, state)
        for epoch in range(len(state["history"]) + 1, config["epochs"] + 1):
            started = time.monotonic()
            atomic_json(output / "status.json", {"state": "training", "arm": arm, "epoch": epoch,
                                                "latest": str(latest), "utc": utc_now()})
            loss = mixed_epoch(model, incumbent.model, optimizer, train, common, extra if arm == "new_field" else None,
                               extra_labels, spec, device, config, epoch, stop)
            if loss is None:
                atomic_json(output / "status.json", {"state": "paused", "arm": arm, "epoch": epoch, "latest": str(latest)})
                return
            result = validate(model, val, dev, classes, spec, device, config, "single_view")
            result.update(epoch=epoch, train_loss=loss, elapsed_seconds=time.monotonic() - started)
            state["history"].append(result)
            weights = cpu_state(model)
            criteria = ("best", "eligible")
            improved = []
            for criterion in criteria:
                if criterion == "eligible" and not is_eligible(result, reference["single_view"], rules):
                    continue
                if choose(result, state[criterion]):
                    state[criterion] = result
                    improved.append(criterion)
            if improved:
                atomic_checkpoint(directory / f"epoch-{epoch:03d}.pt", {"weights": weights, "result": result})
            state.update(weights=weights, optimizer=optimizer.state_dict(), rng=rng_state())
            atomic_checkpoint(latest, state)
            atomic_json(directory / "history.json", state["history"])
            print(arm, epoch, result["master"], "field", result["selection_field_f1"], flush=True)
            if stop["requested"]:
                return
        milestone = state["eligible"] or state["best"]
        path = directory / f"epoch-{milestone['epoch']:03d}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["weights"], strict=True)
        d4_path = directory / "d4.json"
        if d4_path.exists():
            metrics = read(d4_path)
            assert metrics["sha256"] == sha256(path)
        else:
            atomic_json(output / "status.json", {"state": "validation_d4", "arm": arm, "utc": utc_now()})
            metrics = validate(model, val, dev, classes, spec, device, config, "d4_probability_mean")
            metrics.update(checkpoint=str(path), sha256=sha256(path), epoch=saved["result"]["epoch"],
                           eligible_with_single_view=bool(state["eligible"]))
            atomic_json(d4_path, metrics)
        selected[arm] = metrics
        print(arm, "D4", metrics["master"], "field", metrics["selection_field_f1"], flush=True)
        if stop["requested"]:
            return
    baseline = reference["d4_probability_mean"]
    control, candidate = selected["matched_control"], selected["new_field"]
    promoted = (is_eligible(candidate, baseline, rules)
                and candidate["selection_field_f1"] >= control["selection_field_f1"] + rules["control_min_gain"])
    atomic_json(output / "decision.json", {"baseline": baseline, "arms": selected, "promotion_guards_passed": promoted,
                                            "test_opened": False, "release": release["fingerprint"],
                                            "development_utc": utc_now(), "base_checkpoint": config["initial_weights"]})
    atomic_json(output / "status.json", {"state": "complete", "utc": utc_now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/leafcheck_field_v2.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / ".trainer.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("This run already has an owner") from error
        run(args.data, args.run, read(args.config), args.resume)


if __name__ == "__main__":
    main()
