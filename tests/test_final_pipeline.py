"""Meaningful contract, metric and exact CPU resume regression checks."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from PIL import Image
import numpy as np
import torch
from leafcheck.common import ROOT, better, metrics_from_matrix
from leafcheck.model import LeafCNN
from leafcheck.preprocessing import PREPROCESSING_VERSION, image_bytes, image_tensor


class FinalPipelineTests(unittest.TestCase):
    def test_custom_architectures_have_177_logits_and_finite_gradients(self):
        torch.set_num_threads(2)
        for architecture in ("plain_v1", "residual_v1"):
            model = LeafCNN(architecture, channels=(4, 8, 16, 32), dropout=0)
            logits = model(torch.randn(3, 3, 32, 32))
            self.assertEqual(tuple(logits.shape), (3, 177))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 100, 176])).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_metrics_include_unpredicted_classes_and_exact_counts(self):
        result = metrics_from_matrix([[2, 1, 0], [0, 1, 0], [0, 1, 0]], ["a", "b", "c"])
        self.assertEqual(result["correct"], 3); self.assertEqual(result["sample_count"], 5)
        self.assertAlmostEqual(result["accuracy"], .6)
        self.assertAlmostEqual(result["macro_f1"], (.8 + .5 + 0) / 3)
        self.assertEqual(result["per_class"][2]["f1"], 0)
        self.assertEqual(sum(row["support"] for row in result["per_class"]), 5)

    def test_selection_uses_macro_f1_then_loss(self):
        reference = {"macro_f1": .4, "loss": 1.0}
        self.assertTrue(better({"macro_f1": .41, "loss": 2.0}, reference))
        self.assertFalse(better({"macro_f1": .39, "loss": .5}, reference))
        self.assertTrue(better({"macro_f1": .4, "loss": .9}, reference))
        self.assertFalse(better(reference, reference))

    def test_orientation_rgb_padding_and_scaling_share_one_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.png"
            image = Image.new("RGB", (5, 9), (20, 80, 140)); exif = Image.Exif(); exif[274] = 6
            image.save(path, exif=exif)
            pixels = np.frombuffer(image_bytes(path, 16), np.uint8).reshape(16, 16, 3)
            self.assertTrue(np.array_equal(pixels[0, 0], [128, 128, 128]))
            self.assertTrue(np.array_equal(pixels[8, 8], [20, 80, 140]))
            spec = {"version": PREPROCESSING_VERSION, "input_size": 16, "mean": [0, 0, 0], "std": [1, 1, 1]}
            tensor = image_tensor(path, spec)
            torch.testing.assert_close(tensor[0, :, 8, 8], torch.tensor([20, 80, 140]) / 255)
            gray = Path(directory) / "gray.png"; Image.new("L", (16, 16), 50).save(gray)
            self.assertEqual(tuple(image_tensor(gray, spec).shape), (1, 3, 16, 16))

    def test_mid_epoch_resume_matches_uninterrupted_cpu_updates(self):
        self.check_resume_device("cpu")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mid_epoch_resume_matches_uninterrupted_mps_updates(self):
        self.check_resume_device("mps")

    def check_resume_device(self, device):
        # Actual caches and augmentation are used; test data is never opened.
        with tempfile.TemporaryDirectory(dir=ROOT / "runs/final177", prefix="resume-check-") as directory:
            directory = Path(directory)
            config = {"seed": 434, "input_size": 128, "device": device, "cpu_threads": 2,
                      "model": {"architecture": "plain_v1", "channels": [4, 8, 16, 32], "num_classes": 177, "dropout": .2},
                      "learning_rate": .002, "weight_decay": .0001, "batch_size": 8,
                      "augmentation": "dihedral", "max_epochs": 2, "min_epochs": 1, "patience": 1}
            path = directory / "config.json"; path.write_text(json.dumps(config))
            base = [sys.executable, "-m", "leafcheck.train"]
            def run(name, updates, resume=False):
                args = base + ["--run", str(directory / name), "--max-updates", str(updates)]
                args += ["--resume"] if resume else ["--config", str(path)]
                subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True, timeout=120)
                return torch.load(directory / name / "last.pt", weights_only=True, map_location="cpu")
            uninterrupted = run("direct", 4)
            run("resumed", 2); resumed = run("resumed", 2, True)
            self.assertEqual(uninterrupted["total_steps"], resumed["total_steps"])
            self.assertEqual(uninterrupted["next_batch"], resumed["next_batch"])
            self.assertEqual(uninterrupted["train_seen"], resumed["train_seen"])
            self.assertAlmostEqual(uninterrupted["train_loss_sum"], resumed["train_loss_sum"], places=4)
            for name, tensor in uninterrupted["model_state_dict"].items():
                tolerance = 0 if device == "cpu" else 1e-6
                torch.testing.assert_close(tensor, resumed["model_state_dict"][name], atol=tolerance, rtol=tolerance)


if __name__ == "__main__": unittest.main()
