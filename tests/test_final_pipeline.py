"""Meaningful contract, metric and exact CPU resume regression checks."""

import json
from hashlib import sha256 as sha256_bytes
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from PIL import Image, ImageOps
import numpy as np
import torch
from leafcheck.common import ROOT, better, metrics_from_matrix, sha256
from leafcheck.model import LeafCNN
from leafcheck.preprocessing import PREPROCESSING_VERSION, image_bytes, image_tensor, prepare_batch


class FinalPipelineTests(unittest.TestCase):
    def test_completed_resume_preserves_the_run(self):
        from leafcheck.train import train
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            status = '{"state":"complete","checkpoint_reload_verified":true}'
            (path / "status.json").write_text(status)
            train({}, path, resume=True)
            self.assertEqual((path / "status.json").read_text(), status)
            self.assertEqual(len(list(path.iterdir())), 1)

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

    def test_input_contract_stays_float32_with_a_different_global_default(self):
        original = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            spec = {"version":PREPROCESSING_VERSION, "mean":[.4,.5,.6], "std":[.2,.2,.3]}
            result = prepare_batch(torch.zeros((2,8,8,3),dtype=torch.uint8),spec)
            self.assertEqual(result.dtype,torch.float32)
        finally:
            torch.set_default_dtype(original)

    def test_orientation_rgb_padding_and_scaling_share_one_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oriented.png"
            image = Image.new("RGB", (5, 9), (20, 80, 140))
            image.putpixel((0, 0), (250, 10, 20)); image.putpixel((1, 1), (10, 250, 20))
            exif = Image.Exif(); exif[274] = 6
            image.save(path, exif=exif)
            pixels = np.frombuffer(image_bytes(path, 16), np.uint8).reshape(16, 16, 3)
            independently_rotated = ImageOps.pad(image.transpose(Image.Transpose.ROTATE_270), (16, 16),
                                                 method=Image.Resampling.BILINEAR, color=(128, 128, 128))
            np.testing.assert_array_equal(pixels, np.asarray(independently_rotated))
            self.assertTrue(np.array_equal(pixels[0, 0], [128, 128, 128]))
            self.assertTrue(np.array_equal(pixels[8, 8], [20, 80, 140]))
            spec = {"version": PREPROCESSING_VERSION, "input_size": 16, "mean": [0, 0, 0], "std": [1, 1, 1]}
            tensor = image_tensor(path, spec)
            torch.testing.assert_close(tensor[0, :, 8, 8], torch.tensor([20, 80, 140]) / 255)
            gray = Path(directory) / "gray.png"; Image.new("L", (16, 16), 50).save(gray)
            self.assertEqual(tuple(image_tensor(gray, spec).shape), (1, 3, 16, 16))

    def test_mid_epoch_resume_matches_uninterrupted_cpu_updates(self):
        self.check_resume_device("cpu")

    def test_modified_pixel_cache_is_rejected_before_training(self):
        from leafcheck.data import PixelCache
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); path = directory / "train.npy"
            np.save(path, np.zeros((2, 8, 8, 3), np.uint8))
            frozen = {"fingerprint": "test-release", "classes": [f"class_{i}" for i in range(177)]}
            metadata = {"dataset_fingerprint": frozen["fingerprint"], "preprocessing_version": PREPROCESSING_VERSION,
                        "pixels_sha256": sha256(path), "count": 2,
                        "rows": [{"class_id": str(i), "class_name": f"class_{i}"} for i in range(2)]}
            (directory / "train.json").write_text(json.dumps(metadata))
            with patch("leafcheck.data.cache_path", return_value=directory):
                self.assertEqual(len(PixelCache("train", 8, frozen)), 2)
                pixels = np.load(path, mmap_mode="r+"); pixels[0, 0, 0, 0] = 1; pixels.flush(); del pixels
                with self.assertRaisesRegex(AssertionError, "integrity"):
                    PixelCache("train", 8, frozen)

    def test_paired_training_addition_changes_only_the_aligned_extra_pixels(self):
        from leafcheck.training_addition import PairedTrainingAddition

        class Base:
            labels = np.array([1, 2, 2])
            pixels = np.stack([np.full((8, 8, 3), i, dtype=np.uint8) for i in (9, 19, 29)])

            def __len__(self):
                return len(self.labels)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = BytesIO()
            Image.new("RGB", (8, 8), (100, 100, 100)).save(original, format="JPEG")
            member = "train/rust/image.jpg"
            archive_path = root / "source.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(member, original.getvalue())
            release = {"base_fingerprint": "frozen", "source_split": "train", "role": "experimental training only",
                       "class_name": "bean__rust", "archive": {"path": archive_path.name, "sha256": sha256(archive_path)},
                       "members": [{"member": member, "class_name": "bean__rust",
                                    "sha256": sha256_bytes(original.getvalue()).hexdigest()}]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(release))
            options = {"manifest": manifest.name, "sha256": sha256(manifest)}
            frozen = {"fingerprint": "frozen", "class_to_idx": {"bean__rust": 2}}
            indices = np.array([3, 0, 2])
            with patch("leafcheck.training_addition.ROOT", root):
                repeat, field = [PairedTrainingAddition(Base(), dict(options, arm=arm), frozen, 8, 434)
                                 for arm in ("matched_repeat", "new_field")]
                repeat_images, repeat_labels = repeat.batch(indices)
                field_images, field_labels = field.batch(indices)
                self.assertEqual(repeat_labels.tolist(), [2, 1, 2])
                self.assertTrue(torch.equal(repeat_labels, field_labels))
                self.assertTrue(torch.equal(repeat_images[1:], field_images[1:]))
                self.assertTrue(torch.equal(field_images[0], torch.full((8, 8, 3), 100, dtype=torch.uint8)))
                self.assertFalse(torch.equal(repeat_images[0], field_images[0]))
                other = BytesIO()
                Image.new("RGB", (8, 8), (150, 150, 150)).save(other, format="JPEG")
                second_member = "train/other/second.jpg"
                with ZipFile(archive_path, "a") as archive:
                    archive.writestr(second_member, other.getvalue())
                release.pop("class_name")
                release["class_names"] = ["bean__rust", "other__healthy"]
                release["archive"]["sha256"] = sha256(archive_path)
                release["members"].insert(0, {"member": second_member, "class_name": "other__healthy",
                                               "sha256": sha256_bytes(other.getvalue()).hexdigest()})
                manifest.write_text(json.dumps(release))
                options["sha256"] = sha256(manifest)
                frozen["class_to_idx"]["other__healthy"] = 1
                repeat = PairedTrainingAddition(Base(), dict(options, arm="matched_repeat"), frozen, 8, 434)
                field = PairedTrainingAddition(Base(), dict(options, arm="new_field"), frozen, 8, 434)
                np.testing.assert_array_equal(repeat.labels, [1, 2, 2, 1, 2])
                np.testing.assert_array_equal(field.labels, repeat.labels)
                self.assertTrue(np.all(repeat.extra[0] == 9))
                self.assertTrue(np.all(field.extra[0] == 150))
                self.assertTrue(np.all(field.extra[1] == 100))
                archive_path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "archive changed"):
                    PairedTrainingAddition(Base(), dict(options, arm="new_field"), frozen, 8, 434)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mid_epoch_resume_matches_uninterrupted_mps_updates(self):
        self.check_resume_device("mps")

    def test_mid_epoch_resume_matches_style_mixing_cpu_updates(self):
        self.check_resume_device("cpu", style_mixing=True)

    def test_feature_style_swaps_statistics_and_leaves_eval_exact(self):
        from leafcheck.feature_style import forward_style, mix_feature_statistics

        values = torch.tensor([[[[0., 2.], [0., 2.]]], [[[10., 14.], [10., 14.]]]], requires_grad=True)
        swapped = mix_feature_statistics(values, torch.tensor([1, 0]), torch.zeros(2, 1, 1, 1))
        torch.testing.assert_close(swapped, values.flip(0))
        swapped.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

        model = LeafCNN("residual_v1", channels=(4, 8, 16, 32)).eval()
        batch = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).view(2, 3, 16, 16) / 1024
        styled = forward_style(model, batch, np.random.default_rng(434), 1.0, 0.1)
        torch.testing.assert_close(styled, model(batch), atol=0, rtol=0)

    def check_resume_device(self, device, style_mixing=False):
        # Actual caches and augmentation are used; test data is never opened.
        with tempfile.TemporaryDirectory(dir=ROOT / "runs/final177", prefix="resume-check-") as directory:
            directory = Path(directory)
            config = {"seed": 434, "input_size": 128, "device": device, "cpu_threads": 2,
                      "model": {"architecture": "plain_v1", "channels": [4, 8, 16, 32], "num_classes": 177, "dropout": .2},
                      "learning_rate": .002, "weight_decay": .0001, "batch_size": 8,
                      "augmentation": "dihedral", "max_epochs": 2, "min_epochs": 1, "patience": 1}
            if style_mixing:
                config["model"]["architecture"] = "residual_v1"
                config.update(feature_style_probability=1.0, feature_style_alpha=0.1)
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
