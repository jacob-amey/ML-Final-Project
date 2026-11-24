"""Load the explicit frozen final-model bundle, independently of training folders."""

import hashlib
import json
import os
from pathlib import Path
import sys
import threading


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


class FinalCNN:
    name = "Final Custom CNN"

    def __init__(self):
        demo = Path(__file__).resolve().parents[1]
        explicit = os.environ.get("LEAFCHECK_FINAL_ARTIFACT")
        if explicit:
            artifact = Path(explicit).expanduser().resolve()
            if not artifact.is_dir():
                raise FileNotFoundError("The selected final model package is unavailable. Restore its files and restart LeafCheck.")
        else:
            reference = demo / "models/final_model.json"
            if not reference.exists():
                raise FileNotFoundError("The final model package is not installed. Restore the saved model bundle and restart LeafCheck.")
            record = json.loads(reference.read_text())
            artifact = (demo / "models" / record["artifact"]).resolve()
            if not artifact.is_relative_to((demo / "models").resolve()):
                raise ValueError("Invalid final-model artifact reference")
        manifest = json.loads((artifact / "artifact.json").read_text())
        if not explicit and (manifest["checkpoint_sha256"] != record["checkpoint_sha256"]
                             or manifest["model_version"] != record["model_version"]):
            raise ValueError("The final-model reference does not identify this bundle")
        if not manifest.get("final_evaluation_complete"):
            raise ValueError("Final model has no completed frozen evaluation")
        if manifest.get("comparison_reference") and not manifest.get("reference_evaluation_complete"):
            raise ValueError("The registered full-data reference evaluation is incomplete")
        if manifest.get("external_protocol_sha256") and not manifest.get("supplementary_evaluation_complete"):
            raise ValueError("The registered supplementary evaluation is incomplete")
        for name, expected in manifest["files"].items():
            path = (artifact / name).resolve()
            if not path.is_relative_to(artifact) or digest(path) != expected:
                raise ValueError("A final-model bundle file does not match its recorded fingerprint")
        sys.path.insert(0, str(artifact / "runtime"))
        from leafcheck.inference import Predictor
        import torch
        torch.set_num_threads(4)
        self.predictor = Predictor(artifact, "cpu")
        self.lock = threading.Lock()
        policy = json.loads(Path(__file__).with_name("reliability_policy.json").read_text())
        self.reliability = None
        if policy["checkpoint_sha256"] == self.predictor.metadata["checkpoint_sha256"]:
            if (policy["screen"] != "max_softmax" or not 0 < policy["threshold"] < 1
                    or policy["inference_policy"] != self.predictor.policy["name"]
                    or policy["class_map_sha256"] != digest(artifact / "class_to_idx.json")
                    or policy["preprocessing_sha256"] != digest(artifact / "preprocessing.json")):
                raise ValueError("The reliability rule does not match the final model contract")
            self.reliability = policy
        self.metadata = {**self.predictor.metadata, "source": "Explicit frozen final-model bundle",
                         "preprocessing_spec": self.predictor.spec,
                         "preprocessing": f"Frame 0, EXIF orientation, RGB, {self.predictor.spec['input_size']} × {self.predictor.spec['input_size']} bilinear gray padding, /255 and training-only RGB standardization"}
        if self.reliability:
            self.metadata["reliability"] = self.reliability
        self.field_router = None
        study = os.environ.get("LEAFCHECK_FIELD_STUDY")
        if study:
            from field_router import FieldRouter
            self.field_router = FieldRouter(self.predictor, study, artifact)
            if not self.reliability or self.field_router.below != self.reliability["threshold"]:
                raise ValueError("The field study requires the original reliability policy")
            self.metadata["experimental_field_router"] = self.field_router.metadata

    def predict(self, path):
        with self.lock:
            result = self.predictor.predict(path)
            if self.field_router:
                result = self.field_router.apply(path, result)
        if self.reliability:
            result["decision"] = "suggestion" if result["score"] >= self.reliability["threshold"] else "uncertain"
        return result
