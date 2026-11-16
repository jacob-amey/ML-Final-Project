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
        else:
            reference = demo / "models/final_model.json"
            if not reference.exists():
                raise FileNotFoundError("The final 177-class model has not been bundled. Run backend/bundle_final.py with its frozen artifact folder.")
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
        self.metadata = {**self.predictor.metadata, "source": "Explicit frozen final-model bundle",
                         "preprocessing_spec": self.predictor.spec,
                         "preprocessing": f"Frame 0, EXIF orientation, RGB, {self.predictor.spec['input_size']} × {self.predictor.spec['input_size']} bilinear gray padding, /255 and training-only RGB standardization"}

    def predict(self, path):
        with self.lock:
            return self.predictor.predict(path)
