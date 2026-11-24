"""Local model adapters. Neither training data nor weights are served to browsers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

from PIL import Image, ImageOps
import torch

from cnn import PlantCNN
from final_model import FinalCNN

DEMO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEMO_ROOT.parent


def newest_checkpoint(run_folder, environment_key, notebook):
    explicit = os.environ.get(environment_key)
    if explicit:
        return Path(explicit).expanduser().resolve()
    bundled = DEMO_ROOT / "models" / run_folder / "best_model.pt"
    if bundled.exists():
        return bundled
    checkpoints = list((PROJECT_ROOT / "runs" / run_folder).glob("*/best_model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"Run {notebook} to save a best_model.pt checkpoint.")
    return max(checkpoints, key=lambda p: p.parent.name)


class SmallCNN:
    name = "Small CNN"
    notebook = "cnn_small_training.ipynb"
    run_folder = "leafcheck_small"
    environment_key = "LEAFCHECK_CNN_CHECKPOINT"
    model_version = "leafcheck-small-v1"

    def __init__(self):
        self.path = newest_checkpoint(self.run_folder, self.environment_key, self.notebook)
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", {})
        if config.get("model_version") != self.model_version:
            raise ValueError(f"This adapter requires a {self.model_version} notebook checkpoint.")
        self.classes = checkpoint["classes"]
        self.input_size = checkpoint["input_size"]
        self.model = PlantCNN(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.lock = threading.Lock()
        self.metadata = {
            "name": self.name, "ready": True, "class_count": len(self.classes),
            "classes": self.classes, "source": self.notebook,
            "model_version": self.model_version,
            "dataset": config.get("dataset", "leafcheck pilot"),
            "train_images": config.get("train_images"),
            "validation_images": config.get("validation_images"),
            "checkpoint": str(self.path.relative_to(PROJECT_ROOT)) if self.path.is_relative_to(PROJECT_ROOT) else self.path.name,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "input_size": self.input_size, "selected_epoch": checkpoint["epoch"],
            "preprocessing": f"EXIF orientation, RGB, {self.input_size} × {self.input_size} bilinear gray padding, pixels / 255",
        }
        metrics_path = self.path.parent / "validation_metrics.json"
        if metrics_path.exists():
            self.metadata["validation"] = json.loads(metrics_path.read_text())
        test_path = self.path.parent / "test_metrics.json"
        if test_path.exists():
            self.metadata["test"] = json.loads(test_path.read_text())
            self.metadata["test_images"] = self.metadata["test"].get("sample_count")

    def predict(self, path):
        with self.lock, torch.inference_mode():
            started = time.perf_counter()
            # Kept identical to load_leaf() in the training notebook.
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = ImageOps.pad(image, (self.input_size, self.input_size),
                                     method=Image.Resampling.BILINEAR, color=(128, 128, 128))
                # Equivalent to torchvision.to_tensor for an RGB byte image.
                # This keeps torchvision out of the app's runtime dependencies.
                pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
                tensor = pixels.reshape(self.input_size, self.input_size, 3)
                tensor = tensor.permute(2, 0, 1).contiguous().float().div(255).unsqueeze(0)
            probabilities = self.model(tensor).softmax(dim=1)[0].tolist()
            index = max(range(len(probabilities)), key=probabilities.__getitem__)
            return {
                "label": self.classes[index], "score": probabilities[index],
                "class_scores": dict(zip(self.classes, probabilities)),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }


class Starter300CNN(SmallCNN):
    """Same architecture and image preparation; a separate set of trained weights."""

    name = "Starter 300 CNN"
    notebook = "cnn_starter300_training.ipynb"
    run_folder = "leafcheck_starter300"
    environment_key = "LEAFCHECK_STARTER300_CHECKPOINT"
    model_version = "leafcheck-starter300-v1"


class CreateML:
    def __init__(self):
        if sys.platform != "darwin":
            raise RuntimeError("Apple Create ML predictions require the local Mac app.")
        configured = os.environ.get("LEAFCHECK_COREML_MODEL")
        bundled = DEMO_ROOT / "models/createml/PlantDataML.mlmodel"
        self.path = Path(configured).expanduser().resolve() if configured else (
            bundled if bundled.exists() else PROJECT_ROOT / "PlantDataML.mlproj/Models/PlantDataML 1.mlmodel"
        )
        if not self.path.exists():
            raise FileNotFoundError("Set LEAFCHECK_COREML_MODEL to your exported .mlmodel or .mlpackage.")
        runtime = DEMO_ROOT / ".runtime"
        runtime.mkdir(exist_ok=True)
        executable = runtime / "coreml-runner"
        source = DEMO_ROOT / "backend/coreml_runner.swift"
        if not executable.exists() or executable.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(["xcrun", "swiftc", "-O", str(source), "-o", str(executable)], check=True, timeout=120)
        self.process = subprocess.Popen([str(executable), str(self.path)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()
        try:
            metadata = self._read(120)
            if not metadata.get("ready"):
                raise RuntimeError(metadata.get("error", "Create ML could not load."))
        except Exception:
            self.close()
            raise
        self.metadata = {**metadata, "name": "Apple Create ML", "source": self.path.name,
                         "preprocessing": "EXIF orientation, Vision center crop and scale"}

    def _read(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError("Create ML prediction timed out. Restart the local app.")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The Core ML process stopped. Restart the local app.")
        return json.loads(line)

    def predict(self, path):
        with self.lock:
            self.process.stdin.write(json.dumps({"image": str(Path(path).resolve())}) + "\n")
            self.process.stdin.flush()
            try:
                result = self._read(60)
            except TimeoutError:
                self.close()
                raise
            if "error" in result:
                raise RuntimeError(result["error"])
            return result

    def close(self):
        if getattr(self, "process", None) and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class ModelComparison:
    model_types = (("final177", FinalCNN), ("cnn", SmallCNN), ("starter300", Starter300CNN), ("createml", CreateML))

    def __init__(self):
        self.adapters = {}
        self.metadata = {}
        for key, adapter in self.model_types:
            try:
                self.adapters[key] = adapter()
                self.metadata[key] = self.adapters[key].metadata
            except Exception as error:
                self.metadata[key] = {"name": getattr(adapter, "name", "Apple Create ML"),
                                      "ready": False, "error": str(error)}
                print(f"{key}: {error}", file=sys.stderr)

    def predict(self, path, crop=None):
        results = {}
        for key, _ in self.model_types:
            if key not in self.adapters:
                results[key] = {"error": self.metadata[key].get("error", "Model unavailable.")}
                continue
            try:
                if key == "final177" and crop is not None:
                    results[key] = self.adapters[key].predict(path, crop=crop)
                else:
                    results[key] = self.adapters[key].predict(path)
            except Exception as error:
                results[key] = {"error": str(error)}
        return {"source": "live", "models": results}

    def close(self):
        for adapter in self.adapters.values():
            if hasattr(adapter, "close"):
                adapter.close()
