"""Small inference runtime; requires only torch and Pillow, never training data."""

import hashlib
import json
from pathlib import Path
import time
import torch
from .model import LeafCNN
from .preprocessing import image_tensor


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Predictor:
    def __init__(self, artifact, device="cpu"):
        artifact = Path(artifact)
        self.directory = artifact if artifact.is_dir() else artifact.parent
        self.path = artifact / "model.pt" if artifact.is_dir() else artifact
        manifest_path = self.directory / "artifact.json"
        self.artifact = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if self.artifact:
            assert file_hash(self.path) == self.artifact["checkpoint_sha256"], "Model artifact fingerprint mismatch"
        checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        assert checkpoint["model_version"] == "leafcheck177-v1"
        self.classes = checkpoint["classes"]; self.spec = checkpoint["preprocessing"]
        assert len(self.classes) == 177 and len(set(self.classes)) == 177
        assert checkpoint["class_to_idx"] == {name: i for i, name in enumerate(self.classes)}
        for name, expected in [("class_to_idx.json", checkpoint["class_to_idx"]), ("preprocessing.json", self.spec),
                               ("config.json", checkpoint["config"])]:
            path = self.directory / name
            if path.exists():
                assert json.loads(path.read_text()) == expected, f"Inconsistent artifact metadata: {name}"
        self.device = torch.device(device)
        self.model = LeafCNN(**checkpoint["model_config"]).to(device=self.device, memory_format=torch.channels_last)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True); self.model.eval()
        self.metadata = {"name": "Final Custom CNN", "ready": True, "class_count": 177, "classes": self.classes,
                         "model_version": "leafcheck177-v1", "dataset": "leafcheck177_final-v1",
                         "dataset_fingerprint": checkpoint["dataset_fingerprint"],
                         "parameters": checkpoint["parameters"], "input_size": self.spec["input_size"],
                         "selected_epoch": checkpoint["epoch"], "train_images": 59008, "validation_images": 16836,
                         "preprocessing": self.spec, "selection": checkpoint["selection"],
                         "validation": checkpoint["validation"], "checkpoint_sha256": file_hash(self.path),
                         "comparison_note": "Earlier pilots have different label spaces/splits; Create ML exposure is not established. Only the declared rule shares this final held-out protocol.",
                         "limitations": "Source annotations; rare classes; missing specimen IDs; unknown-input and field-photo performance not guaranteed."}
        evaluation = self.directory / "test_metrics.json"
        if evaluation.exists():
            self.metadata["test"] = json.loads(evaluation.read_text()); self.metadata["test_images"] = 8425
        baseline = self.directory / "baseline_metrics.json"
        if baseline.exists(): self.metadata["baseline"] = json.loads(baseline.read_text())

    @torch.inference_mode()
    def logits(self, path):
        return self.model(image_tensor(path, self.spec, self.device)).cpu()

    @torch.inference_mode()
    def predict(self, path):
        start = time.perf_counter()
        probabilities = self.logits(path).softmax(1)[0].tolist()
        index = max(range(177), key=probabilities.__getitem__)
        return {"label": self.classes[index], "score": probabilities[index],
                "class_scores": dict(zip(self.classes, probabilities)),
                "elapsed_ms": (time.perf_counter() - start) * 1000,
                "model_version": self.metadata["model_version"],
                "score_interpretation": "Softmax output score, not a calibrated probability of correctness."}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True); parser.add_argument("image", type=Path)
    args = parser.parse_args(); torch.set_num_threads(4)
    result = Predictor(args.artifact).predict(args.image)
    result["top_five"] = sorted(result.pop("class_scores").items(), key=lambda item: -item[1])[:5]
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
