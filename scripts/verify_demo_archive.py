"""Extract the delivery archive and rehearse a fresh, dataset-free real process."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leafcheck.common import hardware


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python",type=Path,default=Path(sys.executable),help="Compatible app-only Python interpreter for the extracted server")
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/LeafCheck-final-demo.zip")
    parser.add_argument("--receipt", type=Path, default=ROOT / "docs/demo_archive.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/archive_verification.json")
    parser.add_argument("--uncertain-photo", type=Path, help="Original JPEG expected to produce an uncertain result")
    parser.add_argument("--potato-photo", type=Path, help="Original JPEG for the user-supplied potato context check")
    args = parser.parse_args()
    # Resolving the executable symlink would leave its virtual environment.
    runtime_python = args.python.absolute()
    runtime_prefix = Path(subprocess.check_output(
        [str(runtime_python), "-c", "import sys; print(sys.prefix)"], text=True).strip())
    assert (runtime_prefix / "bin/python").is_file() and (runtime_prefix / "pyvenv.cfg").is_file(), \
        "Pass the app-only virtual environment's bin/python"
    archive = args.archive
    record = json.loads(args.receipt.read_text())
    assert digest(archive) == record["sha256"]
    with tempfile.TemporaryDirectory(prefix="leafcheck-delivery-") as temporary:
        directory = Path(temporary).resolve()
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            assert all((directory / name).resolve().is_relative_to(directory) for name in names)
            assert not any("notebook" in name.lower() or name.endswith(".ipynb") or "/data/" in name for name in names)
            bundle.extractall(directory)
        demo = directory / "LeafCheck"
        manifest = json.loads((demo / "PACKAGE_MANIFEST.json").read_text())
        assert len(names) == len(manifest) + 1
        assert all(digest(demo / name) == expected for name, expected in manifest.items())
        (demo / ".venv").symlink_to(runtime_prefix, target_is_directory=True)
        launcher = demo / "Launch LeafCheck.command"
        launcher.chmod(0o755)
        reference = json.loads((demo / "models/final_model.json").read_text())
        artifact = demo / "models" / reference["artifact"]
        examples = json.loads((artifact / "demo_examples.json").read_text())
        environment = os.environ.copy()
        for key in ["LEAFCHECK_CNN_CHECKPOINT", "LEAFCHECK_STARTER300_CHECKPOINT", "LEAFCHECK_COREML_MODEL",
                    "LEAFCHECK_FINAL_ARTIFACT", "LEAFCHECK_FIELD_STUDY", "PYTHONPATH"]:
            environment.pop(key, None)
        environment["LEAFCHECK_OPEN_BROWSER"] = "0"
        log = directory / "server.log"
        with log.open("w") as output:
            launched = time.perf_counter()
            process = subprocess.Popen([str(launcher), "--port", "0"],
                                       cwd=demo, stdout=output, stderr=subprocess.STDOUT, env=environment)
            try:
                address = None
                while time.perf_counter() - launched < 45 and process.poll() is None:
                    match = re.search(r"LeafCheck is ready: (http://127\.0\.0\.1:\d+)", log.read_text())
                    if match:
                        address = match.group(1); break
                    time.sleep(.2)
                assert address, log.read_text()
                startup_ms = (time.perf_counter() - launched) * 1000
                with urlopen(address + "/api/status", timeout=10) as response: status = json.load(response)
                assert set(status["models"]) == {"final177", "cnn", "starter300", "createml"}
                assert all(model["ready"] for model in status["models"].values())
                assert status["models"]["final177"]["checkpoint_sha256"] == reference["checkpoint_sha256"]
                measurements = []; observed = []
                policy = status["models"]["final177"].get("reliability")
                for i in range(14):
                    row = examples[i % len(examples)]
                    raw = (artifact / row["sample_file"]).read_bytes()
                    start = time.perf_counter()
                    with urlopen(Request(address + "/api/predict", data=raw, headers={"Content-Type":"image/jpeg"}), timeout=45) as response:
                        result = json.load(response)
                    elapsed = (time.perf_counter() - start) * 1000
                    final = result["models"]["final177"]
                    assert final["label"] == row["predicted_class"]
                    assert abs(final["score"] - row["score"]) < 1e-4
                    assert all("label" in prediction for prediction in result["models"].values())
                    if policy:
                        assert final["decision"] == ("suggestion" if final["score"] >= policy["threshold"] else "uncertain")
                    if i >= 2: measurements.append(elapsed)
                    if i < 2:
                        observed.append({"purpose":row["purpose"], "reference_label":row["actual_class"],
                                         "predicted_label":final["label"], "score":final["score"],
                                         "correct":row["actual_class"] == final["label"], "all_four_predicted":True})
                for raw, mime, code in [(b"not an image", "image/jpeg", 400), (b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", "image/gif", 415)]:
                    # A malformed GIF may fail decode before its format check; use a valid source below.
                    if mime == "image/gif":
                        from io import BytesIO
                        from PIL import Image
                        buffer = BytesIO(); Image.new("RGB", (8, 8)).save(buffer, "GIF"); raw = buffer.getvalue()
                    try:
                        urlopen(Request(address + "/api/predict", data=raw, headers={"Content-Type":mime}), timeout=10)
                    except HTTPError as error:
                        assert error.code == code; error.close()
                    else: raise AssertionError("Invalid/unsupported upload was accepted")
                evidence = {"archive_sha256":record["sha256"], "checkpoint_sha256":reference["checkpoint_sha256"],
                            "hardware":hardware(),
                            "fresh_process":True, "working_directory_has_training_data":False, "notebooks_present":False,
                            "manifest_files_verified":len(manifest), "all_four_real_models_ready":True,
                            "python_executable":str(runtime_python), "python_venv":str(runtime_prefix),
                            "startup_command":"./Launch LeafCheck.command --port 0 (automatic browser disabled for the HTTP check)",
                            "startup_ms_including_first_swift_compile":startup_ms,
                            "http_latency":{"warmup_requests":2,"timed_requests":len(measurements),
                                            "p50_ms":float(np.median(measurements)), "p95_ms":float(np.percentile(measurements,95)),
                                            "measurements_ms":measurements,
                                            "conditions":"Loopback HTTP, alternating packaged success/mistake, all four real models, warm local files, includes image upload/validation and JSON response; not CNN-only latency"},
                            "example_results":observed, "invalid_image_http_status":400,"unsupported_format_http_status":415}
                if args.uncertain_photo:
                    assert policy is not None
                    request = Request(address + "/api/predict", data=args.uncertain_photo.read_bytes(),
                                      headers={"Content-Type": "image/jpeg"})
                    with urlopen(request, timeout=20) as response:
                        prediction = json.load(response)["models"]["final177"]
                    assert prediction["decision"] == "uncertain"
                    evidence["uncertain_photo"] = {"sha256": digest(args.uncertain_photo),
                                                   "model_score": prediction["score"],
                                                   "verified_decision": prediction["decision"]}
                if args.potato_photo:
                    photo = args.potato_photo.read_bytes()
                    outputs = []
                    for query in ("", "?crop=potato"):
                        request = Request(address + "/api/predict" + query, data=photo,
                                          headers={"Content-Type": "image/jpeg"})
                        with urlopen(request, timeout=30) as response:
                            outputs.append(json.load(response)["models"]["final177"])
                    automatic, known = outputs
                    labels = status["models"]["final177"]["crop_context"]["potato"]
                    assert len(automatic["class_scores"]) == 177 and list(known["class_scores"]) == labels
                    assert known["scope"] == "known_potato"
                    subtotal = sum(automatic["class_scores"][label] for label in labels)
                    drift = max(abs(known["class_scores"][label] - automatic["class_scores"][label] / subtotal)
                                for label in labels)
                    assert drift <= 1e-5
                    assert known["label"] == max(known["class_scores"], key=known["class_scores"].get)
                    evidence["known_potato_upload"] = {"sha256": digest(args.potato_photo),
                                                       "automatic_label": automatic["label"],
                                                       "known_crop_label": known["label"],
                                                       "score_difference": drift,
                                                       "known_crop_scores": known["class_scores"]}
                args.output.write_text(json.dumps(evidence, indent=2) + "\n")
                print(json.dumps(evidence, indent=2))
            finally:
                if process.poll() is None: process.send_signal(signal.SIGINT)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()


if __name__ == "__main__": main()
