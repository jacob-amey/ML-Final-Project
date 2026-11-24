"""Real final-model API parity and artifact/format contracts. No training data needed."""

from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
import torch

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO / "backend"))
from final_model import FinalCNN

BASE = os.environ.get("LEAFCHECK_TEST_URL", "http://127.0.0.1:8765")


class FinalModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = FinalCNN()

    def request(self, raw, content_type="image/png"):
        with urlopen(Request(BASE + "/api/predict", data=raw, headers={"Content-Type": content_type}), timeout=30) as response:
            return json.load(response)["models"]["final177"]

    def assert_parity(self, raw, suffix=".png"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ("image" + suffix); path.write_bytes(raw)
            expected = self.adapter.predict(path); actual = self.request(raw)
        self.assertEqual(actual["label"], expected["label"])
        self.assertEqual(len(actual["class_scores"]), 177)
        self.assertAlmostEqual(sum(actual["class_scores"].values()), 1, places=5)
        self.assertEqual(list(actual["class_scores"]), self.adapter.predictor.classes)
        difference = max(abs(actual["class_scores"][key] - value) for key, value in expected["class_scores"].items())
        self.assertLessEqual(difference, 1e-5)
        self.assertTrue(self.adapter.predictor.model.training is False)

    def test_frozen_artifact_and_saved_rule_evaluation(self):
        with urlopen(BASE + "/api/status", timeout=10) as response: status = json.load(response)
        final = status["models"]["final177"]
        self.assertTrue(final["ready"]); self.assertEqual(final["class_count"], 177)
        self.assertEqual(final["train_images"], 59008)
        self.assertEqual(final["test"]["sample_count"], 8425)
        self.assertEqual(final["baseline"]["sample_count"], 8425)
        self.assertEqual(final["reference"]["sample_count"], 8425)
        self.assertEqual(final["reference"]["class_count"], 177)
        self.assertNotEqual(final["reference"]["checkpoint_sha256"], final["checkpoint_sha256"])
        self.assertEqual(final["checkpoint_sha256"], self.adapter.metadata["checkpoint_sha256"])

    def test_real_packaged_success_and_mistake_match_standalone(self):
        directory = self.adapter.predictor.directory
        records = json.loads((directory / "demo_examples.json").read_text())
        self.assertEqual({row["purpose"] for row in records}, {"success", "mistake"})
        for row in records:
            self.assert_parity((directory / row["sample_file"]).read_bytes(), Path(row["sample_file"]).suffix)
            result = self.request((directory / row["sample_file"]).read_bytes())
            self.assertEqual(result["label"], row["predicted_class"])
            self.assertAlmostEqual(result["score"], row["score"], places=4)

    def test_jpeg_exif_grayscale_and_rgba_uploads(self):
        cases = []
        image = Image.new("RGB", (73, 121), (25, 130, 45))
        for x in range(30):
            for y in range(45): image.putpixel((x, y), (200, 30, 10))
        exif = Image.Exif(); exif[274] = 6
        buffer = BytesIO(); image.save(buffer, "JPEG", exif=exif); cases.append((buffer.getvalue(), ".jpg"))
        for mode, color in [("L", 90), ("RGBA", (40, 90, 140, 150))]:
            buffer = BytesIO(); Image.new(mode, (91, 55), color).save(buffer, "PNG"); cases.append((buffer.getvalue(), ".png"))
        for raw, suffix in cases:
            with self.subTest(suffix=suffix, bytes=len(raw)): self.assert_parity(raw, suffix)

    def test_explicit_missing_bundle_has_clear_error(self):
        from unittest.mock import patch
        with patch.dict(os.environ, {"LEAFCHECK_FINAL_ARTIFACT": "/nonexistent/leafcheck-final-model"}):
            with self.assertRaises(FileNotFoundError): FinalCNN()

    def test_valid_jpeg_above_pixel_cap_reports_size_limit(self):
        buffer = BytesIO()
        Image.new("RGB", (6000, 4001), (25, 130, 45)).save(buffer, "JPEG")
        with self.assertRaises(HTTPError) as error:
            self.request(buffer.getvalue(), "image/jpeg")
        with error.exception as response:
            self.assertEqual(response.code, 413)
            self.assertIn("24 megapixels", json.load(response)["error"])

    def test_multi_picture_jpeg_uses_frame_zero(self):
        directory = self.adapter.predictor.directory
        fixtures = json.loads((directory / "integration_fixtures.json").read_text())
        self.assertTrue(fixtures)
        for fixture in fixtures:
            path = directory / fixture["sample_file"]
            with Image.open(path) as image:
                self.assertEqual(image.format, "MPO"); self.assertGreaterEqual(image.n_frames, 2)
            self.assert_parity(path.read_bytes(), ".jpg")

    def test_one_missing_model_preserves_other_live_predictions(self):
        from unittest.mock import patch
        from models import ModelComparison
        with patch.dict(os.environ, {"LEAFCHECK_FINAL_ARTIFACT": "/nonexistent/leafcheck-final-model"}):
            comparison = ModelComparison()
        try:
            self.assertFalse(comparison.metadata["final177"]["ready"])
            results = comparison.predict(DEMO / "web/assets/healthy.jpg")
            self.assertIn("error", results["models"]["final177"])
            for key in ("cnn", "starter300", "createml"):
                self.assertIn("label", results["models"][key])
        finally:
            comparison.close()


if __name__ == "__main__": unittest.main()
