"""Trace labeled development fixtures through the real local upload handler."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from PIL import Image, ImageOps
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leafcheck.common import atomic_json
from leafcheck.inference import Predictor
from leafcheck.preprocessing import image_bytes, image_tensor


def observe(output):
    sys.path.insert(0, str(ROOT / "leafcheck-demo/backend"))
    import server
    original = server.ModelComparison.predict

    def capture(self, path):
        output.write_bytes(path.read_bytes())
        return original(self, path)

    server.ModelComparison.predict = capture
    sys.argv = [__file__, "--port", "0"]
    server.main()


def upload(url, path, mime):
    with urlopen(Request(url + "/api/predict", data=path.read_bytes(),
                        headers={"Content-Type": mime}), timeout=30) as response:
        return json.load(response)["models"]["final177"]


def app_status(url):
    with urlopen(url + "/api/status", timeout=10) as response:
        return json.load(response)


def error_response(url, path, mime):
    try:
        upload(url, path, mime)
    except HTTPError as error:
        with error:
            return {"status": error.code, **json.load(error)}
    raise AssertionError(f"{path.name} was unexpectedly accepted")


def trace(url, cases, output, previous_url=None):
    artifact = ROOT / "artifacts/leafcheck177-v1"
    reference = json.loads((ROOT / "leafcheck-demo/models/final_model.json").read_text())
    torch.set_num_threads(4)
    predictor = Predictor(artifact)
    status = app_status(url)
    live = status["models"]["final177"]
    assert live["checkpoint_sha256"] == reference["checkpoint_sha256"] == predictor.metadata["checkpoint_sha256"]
    assert live["classes"] == predictor.classes and not predictor.model.training and not predictor.scoring_model.training
    assert live["preprocessing_spec"] == predictor.spec
    previous_status = app_status(previous_url) if previous_url else None
    if previous_status:
        assert previous_status["models"]["final177"]["checkpoint_sha256"] == live["checkpoint_sha256"]
    evidence = []
    with tempfile.TemporaryDirectory(prefix="leafcheck-trace-") as folder:
        intercepted = Path(folder) / "upload.png"
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--observe", str(intercepted)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            startup = []
            for line in process.stdout:
                startup.append(line)
                address = re.search(r"LeafCheck is ready: (http://127[.]0[.]0[.]1:[0-9]+)", line)
                if address:
                    instrumented = address[1]
                    break
            else:
                raise RuntimeError("Upload trace server failed to start\n" + "".join(startup))
            for case in cases:
                path = ROOT / case["path"]
                raw = path.read_bytes()
                with Image.open(BytesIO(raw)) as source:
                    size = list(source.size)
                    mode = source.mode
                    orientation = source.getexif().get(274, 1)
                    mime = "image/png" if source.format == "PNG" else "image/jpeg"
                    rgb = ImageOps.exif_transpose(source).convert("RGB")
                if case.get("expect_pixel_limit"):
                    blocked = error_response(url, path, mime)
                    assert blocked["status"] == 413 and "24 megapixels" in blocked["error"]
                    previous = error_response(previous_url, path, mime) if previous_url else None
                    if previous_url:
                        assert previous["status"] == 400
                    evidence.append({**case, "input_sha256": sha256(raw).hexdigest(),
                                     "source_size": size, "previous_api_response": previous,
                                     "api_response": blocked})
                    continue
                real = upload(url, path, mime)
                captured = upload(instrumented, path, mime)
                with Image.open(intercepted) as received:
                    assert received.mode == "RGB" and received.size == rgb.size
                    assert received.tobytes() == rgb.tobytes()
                    assert received.getexif().get(274, 1) == 1
                side = predictor.spec["input_size"]
                direct = image_bytes(path, side)
                assert direct == image_bytes(intercepted, side)
                raw_tensor = image_tensor(path, predictor.spec)
                app_tensor = image_tensor(intercepted, predictor.spec)
                with torch.inference_mode():
                    raw_logit_diff = (predictor.model(raw_tensor) - predictor.model(app_tensor)).abs().max().item()
                    policy_logit_diff = (predictor.logits(path) - predictor.logits(intercepted)).abs().max().item()
                tensor_diff = (raw_tensor - app_tensor).abs().max().item()
                standalone = predictor.predict(path)
                assert standalone["label"] == captured["label"] == real["label"]
                max_score_diff = 0.0
                for actual in (real, captured):
                    assert list(actual["class_scores"]) == predictor.classes
                    max_score_diff = max(max_score_diff, max(
                        abs(actual["class_scores"][name] - standalone["class_scores"][name])
                        for name in predictor.classes))
                assert tensor_diff == raw_logit_diff == policy_logit_diff == 0
                assert max_score_diff <= 1e-5
                if previous_url:
                    previous = upload(previous_url, path, mime)
                    assert previous["label"] == real["label"] and previous["class_scores"] == real["class_scores"]
                evidence.append({
                    **case, "input_sha256": sha256(raw).hexdigest(), "source_size": size, "source_mode": mode,
                    "exif_orientation": orientation, "corrected_rgb_size": list(rgb.size),
                    "matched_decoded_rgb_sha256": sha256(rgb.tobytes()).hexdigest(),
                    "matched_resize_sha256": sha256(direct).hexdigest(), "tensor_shape": list(raw_tensor.shape),
                    "max_tensor_difference": tensor_diff, "max_raw_logit_difference": raw_logit_diff,
                    "max_policy_logit_difference": policy_logit_diff, "max_api_score_difference": max_score_diff,
                    "score_count": len(real["class_scores"]), "label": real["label"],
                    "matches_previous_app": True if previous_url else None,
                    "label_index": predictor.classes.index(real["label"]), "score": real["score"],
                    "top_three": sorted(real["class_scores"].items(), key=lambda item: -item[1])[:3],
                })
        finally:
            process.terminate()
            process.communicate(timeout=10)
    atomic_json(output, {"normal_app_revision": status["app_revision"],
                         "previous_app_revision": previous_status["app_revision"] if previous_status else None,
                         "hash": live["checkpoint_sha256"],
                         "preprocessing": predictor.spec, "cases": evidence})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous-url")
    parser.add_argument("--observe", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.observe:
        observe(args.observe)
    elif not args.cases or not args.output:
        parser.error("--cases and --output are required")
    else:
        trace(args.url, json.loads(args.cases.read_text()), args.output, args.previous_url)


if __name__ == "__main__":
    main()
