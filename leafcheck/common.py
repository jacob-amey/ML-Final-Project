"""Small, explicit metric and artifact utilities."""

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/leafcheck177_final"
RUNS = ROOT / "runs/final177"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.flush(); os.fsync(file.fileno())
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file:
        torch.save(value, file); file.flush(); os.fsync(file.fileno())
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def environment(device):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "leafcheck").glob("*.py"))}
    return {"created_utc": utc_now(), "python": sys.version, "executable": sys.executable,
            "torch": str(torch.__version__), "numpy": np.__version__, "pillow": Image.__version__,
            "platform": platform.platform(), "device": str(device), "git_revision": revision,
            "source_sha256": sources,
            "source_fingerprint": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()}


def metrics_from_matrix(matrix, classes, loss=None):
    matrix = np.asarray(matrix, dtype=np.int64)
    tp = matrix.diagonal().astype(np.float64)
    support = matrix.sum(axis=1); predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) != 0)
    total = int(support.sum())
    return {"accuracy": float(tp.sum() / total) if total else 0.0, "macro_f1": float(f1.mean()),
            "correct": int(tp.sum()), "sample_count": total, "loss": loss,
            "confusion_matrix": matrix.tolist(),
            "per_class": [{"class_index": i, "class_name": name, "precision": float(precision[i]),
                           "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
                          for i, name in enumerate(classes)]}


def update_matrix(matrix, targets, predictions):
    count = len(matrix)
    matrix += np.bincount(np.asarray(targets) * count + np.asarray(predictions), minlength=count * count).reshape(count, count)


def better(candidate, reference):
    if reference is None:
        return True
    difference = candidate["macro_f1"] - reference["macro_f1"]
    return difference > 1e-6 or (abs(difference) <= 1e-6 and candidate["loss"] < reference["loss"] - 1e-10)
