"""Paired external training additions using the original split's preprocessing."""

from hashlib import sha256 as hash_bytes
from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import torch

from .common import ROOT, sha256
from .preprocessing import image_bytes


class PairedTrainingAddition:
    """Expose the base cache plus a matching-label suffix in the same batch order."""

    def __init__(self, base, options, frozen, size, seed):
        self.base = base
        self.size = size
        manifest_path = ROOT / options["manifest"]
        if sha256(manifest_path) != options["sha256"]:
            raise ValueError("Training addition manifest changed")
        release = json.loads(manifest_path.read_text())
        if (release["base_fingerprint"] != frozen["fingerprint"] or release["source_split"] != "train"
                or release["role"] != "experimental training only"):
            raise ValueError("Training addition must match the frozen base and a training-only role")
        members = release["members"]
        labels = release["class_names"] if "class_names" in release else [release["class_name"]]
        if (not members or len({m["sha256"] for m in members}) != len(members)
                or len({m["member"] for m in members}) != len(members)
                or not labels or len(set(labels)) != len(labels)
                or set(labels) != {m["class_name"] for m in members}
                or not set(labels).issubset(frozen["class_to_idx"])
                or any(not m["member"].startswith("train/") for m in members)):
            raise ValueError("Invalid training addition membership")
        targets = np.array([frozen["class_to_idx"][row["class_name"]] for row in members], dtype=np.int64)
        self.labels = np.concatenate((base.labels, targets))
        if options["arm"] == "matched_repeat":
            rng = np.random.default_rng([seed, 972])
            draws = np.empty(len(targets), dtype=np.int64)
            for target in np.unique(targets):
                available = np.flatnonzero(base.labels == target)
                if not len(available):
                    raise ValueError("No eligible master rows for the repeat arm")
                selected = targets == target
                draws[selected] = rng.choice(available, size=int(selected.sum()), replace=True)
            self.extra = np.array(base.pixels[draws], copy=True)
        elif options["arm"] == "new_field":
            path = ROOT / release["archive"]["path"]
            if sha256(path) != release["archive"]["sha256"]:
                raise ValueError("Training archive changed")
            self.extra = np.empty((len(members), size, size, 3), dtype=np.uint8)
            with ZipFile(path) as archive:
                for index, row in enumerate(members):
                    content = archive.read(row["member"])
                    if hash_bytes(content).hexdigest() != row["sha256"]:
                        raise ValueError(f"Training image changed: {row['member']}")
                    raw = image_bytes(BytesIO(content), size)
                    self.extra[index] = np.frombuffer(raw, dtype=np.uint8).reshape(size, size, 3)
        else:
            raise ValueError("Unknown paired training arm")

    def __len__(self):
        return len(self.labels)

    def batch(self, indices):
        in_base = indices < len(self.base)
        pixels = np.empty((len(indices), self.size, self.size, 3), dtype=np.uint8)
        pixels[in_base] = self.base.pixels[indices[in_base]]
        pixels[~in_base] = self.extra[indices[~in_base] - len(self.base)]
        return torch.from_numpy(pixels), torch.from_numpy(self.labels[indices].copy())
