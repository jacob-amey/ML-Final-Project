"""Opt-in inference for the independently frozen field adaptation study."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch


class FieldRouter:
    def __init__(self, predictor, study, artifact):
        from leafcheck.inference import file_hash
        from leafcheck.scoring import PredictionScores

        directory = Path(study).expanduser().resolve()
        bundle = json.loads((directory / "bundle.json").read_text())
        files = ("freeze.json", "metrics.json", "model.pt")
        if any(file_hash(directory / name) != bundle["files_sha256"][name] for name in files):
            raise ValueError("The experimental field study package is inconsistent.")
        freeze = json.loads((directory / "freeze.json").read_text())
        metrics = json.loads((directory / "metrics.json").read_text())
        policy = freeze["experimental_routing"]
        if (not freeze["deployment_guards_passed_on_validation"]
                or freeze["reference_sha256"] != predictor.metadata["checkpoint_sha256"]
                or freeze["policy"] != predictor.policy["name"]
                or freeze["class_map_sha256"] != file_hash(artifact / "class_to_idx.json")
                or freeze["preprocessing_sha256"] != file_hash(artifact / "preprocessing.json")
                or freeze["candidate_sha256"] != bundle["files_sha256"]["model.pt"]
                or metrics["freeze_sha256"] != bundle["files_sha256"]["freeze.json"]):
            raise ValueError("The field study does not match the active model and evaluation.")
        self.below = policy["reference_below"]
        self.at_least = policy["candidate_at_least"]
        if not 0 < self.below < self.at_least < 1:
            raise ValueError("Invalid field study score limits.")
        with (directory / "bundle.json").open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        model = deepcopy(predictor.model)
        weights = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights["weights"], strict=True)
        self.scorer = PredictionScores(model, predictor.policy).eval()
        self.predictor = predictor
        self.metadata = {
            "version": bundle["version"], "fingerprint": fingerprint,
            "candidate_checkpoint_sha256": freeze["candidate_sha256"],
            "reference_below": self.below, "candidate_at_least": self.at_least,
        }

    @torch.inference_mode()
    def apply(self, path, result):
        from leafcheck.preprocessing import image_tensor

        if result["score"] >= self.below:
            return {**result, "score_origin": "original_cnn"}
        started = time.perf_counter()
        x = image_tensor(path, self.predictor.spec, self.predictor.device)
        scores = self.scorer(x).softmax(1)[0].tolist()
        winner = max(range(len(scores)), key=scores.__getitem__)
        elapsed = result["elapsed_ms"] + (time.perf_counter() - started) * 1000
        if scores[winner] < self.at_least:
            return {**result, "elapsed_ms": elapsed, "score_origin": "original_cnn"}
        label = self.predictor.classes[winner]
        return {**result, "label": label, "score": scores[winner], "elapsed_ms": elapsed,
                "class_scores": dict(zip(self.predictor.classes, scores)), "score_origin": "field_candidate"}
