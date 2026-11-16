"""Run against the local app; verifies actual saved model behavior, not mocks."""

import ast
from io import BytesIO
import json
import os
from pathlib import Path
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor

DEMO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = DEMO_ROOT.parent
sys.path.insert(0, str(DEMO_ROOT / "backend"))
from models import SmallCNN, Starter300CNN

BASE = os.environ.get("LEAFCHECK_TEST_URL", "http://127.0.0.1:8765")


class IntegrationTests(unittest.TestCase):
    def request(self, path, body=None, content_type="image/jpeg"):
        request = Request(BASE + path, data=body,
                          headers={"Content-Type": content_type} if body is not None else {})
        with urlopen(request, timeout=90) as response:
            return response.status, response.read()

    def assert_checkpoint_reference(self, adapter, notebook_name, validation_path):
        notebook = json.loads((PROJECT_ROOT / notebook_name).read_text())
        source = next("".join(cell["source"]) for cell in notebook["cells"]
                      if cell["cell_type"] == "code" and "def load_leaf(" in "".join(cell["source"]))
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == "load_leaf")
        context = {"Image": Image, "ImageOps": ImageOps, "to_tensor": to_tensor,
                   "INPUT_SIZE": adapter.input_size}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook_loader", "exec"), context)
        dataset = ImageFolder(PROJECT_ROOT / validation_path,
                              loader=context["load_leaf"])
        images, _ = next(iter(DataLoader(dataset, batch_size=16, shuffle=False)))
        checkpoint = torch.load(adapter.path, weights_only=True, map_location="cpu")
        with torch.inference_mode():
            logits = adapter.model(images)
        torch.testing.assert_close(logits, checkpoint["reference_logits"], rtol=1e-4, atol=1e-5)

    def test_original_cnn_matches_notebooks_saved_reference_logits(self):
        self.assert_checkpoint_reference(SmallCNN(), "cnn_small_training.ipynb",
                                         "data/processed/leafcheck/validation")

    def test_starter300_matches_notebooks_saved_reference_logits(self):
        self.assert_checkpoint_reference(Starter300CNN(), "cnn_starter300_training.ipynb",
                                         "Plant_CNN_Starter_300/val")

    def test_cnn_experiments_cannot_load_each_others_weights(self):
        original = SmallCNN()
        starter = Starter300CNN()
        self.assertNotEqual(original.path, starter.path)
        with patch.dict(os.environ, {"LEAFCHECK_STARTER300_CHECKPOINT": str(original.path)}):
            with self.assertRaisesRegex(ValueError, "leafcheck-starter300-v1"):
                Starter300CNN()
        with patch.dict(os.environ, {"LEAFCHECK_CNN_CHECKPOINT": str(starter.path)}):
            with self.assertRaisesRegex(ValueError, "leafcheck-small-v1"):
                SmallCNN()

    def test_all_three_real_models_and_regression_examples(self):
        _, raw_status = self.request("/api/status")
        status = json.loads(raw_status)
        self.assertEqual(status["mode"], "live")
        self.assertTrue(status["app_id"].startswith("leafcheck-local-"))
        self.assertTrue(status["models"]["cnn"]["ready"])
        self.assertTrue(status["models"]["starter300"]["ready"])
        self.assertTrue(status["models"]["createml"]["ready"])
        self.assertEqual(status["models"]["cnn"]["class_count"], 3)
        self.assertEqual(status["models"]["starter300"]["class_count"], 3)
        self.assertEqual(status["models"]["createml"]["class_count"], 177)
        self.assertEqual(status["models"]["starter300"]["train_images"], 210)
        self.assertEqual(status["models"]["starter300"]["validation_images"], 60)
        self.assertEqual(status["models"]["starter300"]["test_images"], 30)
        self.assertEqual(status["models"]["starter300"]["source"], "cnn_starter300_training.ipynb")
        examples = json.loads((DEMO_ROOT / "tests/fixtures/predictions.json").read_text())
        for sample, expected in examples.items():
            with self.subTest(sample=sample):
                raw = (DEMO_ROOT / "web/assets" / f"{sample}.jpg").read_bytes()
                _, body = self.request("/api/predict", raw)
                result = json.loads(body)
                self.assertEqual(result["source"], "live")
                self.assertEqual(set(result["models"]), {"final177", "cnn", "starter300", "createml"})
                for key, count in [("cnn", 3), ("starter300", 3), ("createml", 177)]:
                    prediction = result["models"][key]
                    recorded = expected[key]
                    self.assertNotIn("error", prediction)
                    self.assertEqual(len(prediction["class_scores"]), count)
                    self.assertAlmostEqual(sum(prediction["class_scores"].values()), 1, places=4)
                    self.assertEqual(prediction["label"], recorded["label"])
                    self.assertAlmostEqual(prediction["score"], recorded["score"], places=4)

    def test_local_browser_files_have_no_external_dependencies(self):
        for path in ["/", "/styles.css", "/app.js", "/assets/leaf.svg"]:
            with self.subTest(path=path), urlopen(BASE + path) as response:
                text = response.read().decode()
                self.assertIn("connect-src 'self'", response.headers["Content-Security-Policy"])
                self.assertIn("font-src 'self'", response.headers["Content-Security-Policy"])
                self.assertNotRegex(text, r"https?://(?!127\.0\.0\.1|localhost|www\.w3\.org)")
                self.assertNotIn("chatgpt", text.lower())
                self.assertNotIn("@import", text)

    def test_app_runs_from_a_copy_without_notebooks_or_datasets(self):
        with tempfile.TemporaryDirectory(prefix="leafcheck-standalone-") as folder:
            copied = Path(folder) / "LeafCheck"
            for name in ["backend", "web", "models"]:
                shutil.copytree(DEMO_ROOT / name, copied / name,
                                ignore=shutil.ignore_patterns("__pycache__"))
            environment = os.environ.copy()
            for key in ["LEAFCHECK_CNN_CHECKPOINT", "LEAFCHECK_STARTER300_CHECKPOINT", "LEAFCHECK_COREML_MODEL", "LEAFCHECK_FINAL_ARTIFACT"]:
                environment.pop(key, None)
            process = subprocess.Popen([sys.executable, str(copied / "backend/server.py"), "--port", "0"],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
            try:
                local_url = None
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline and process.poll() is None:
                    if select.select([process.stdout], [], [], .2)[0]:
                        line = process.stdout.readline()
                        match = re.search(r"LeafCheck is ready: (http://127\.0\.0\.1:\d+)", line)
                        if match:
                            local_url = match.group(1)
                            break
                self.assertIsNotNone(local_url, "The standalone copy did not start.")
                with urlopen(local_url + "/api/status", timeout=10) as response:
                    status = json.load(response)
                self.assertEqual(set(status["models"]), {"final177", "cnn", "starter300", "createml"})
                self.assertTrue(all(model["ready"] for model in status["models"].values()), status)
                raw = (copied / "web/assets/healthy.jpg").read_bytes()
                request = Request(local_url + "/api/predict", data=raw, headers={"Content-Type": "image/jpeg"})
                with urlopen(request, timeout=30) as response:
                    result = json.load(response)
                self.assertEqual(result["models"]["starter300"]["label"], "tomato__healthy")
                self.assertNotIn("error", result["models"]["cnn"])
                self.assertNotIn("error", result["models"]["createml"])
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    def test_invalid_image_rejected(self):
        with self.assertRaises(HTTPError) as error:
            self.request("/api/predict", b"not an image")
        self.assertEqual(error.exception.code, 400)
        error.exception.close()

    def test_unsupported_image_rejected(self):
        buffer = BytesIO()
        Image.new("RGB", (8, 8)).save(buffer, format="GIF")
        with self.assertRaises(HTTPError) as error:
            self.request("/api/predict", buffer.getvalue(), "image/gif")
        self.assertEqual(error.exception.code, 415)
        error.exception.close()

    def test_weights_and_source_are_not_served(self):
        for path in ["/../backend/models.py", "/%2e%2e/backend/models.py", "/.openai/hosting.json", "/assets/"]:
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                self.request(path)
            self.assertEqual(error.exception.code, 404)
            error.exception.close()


if __name__ == "__main__":
    unittest.main()
