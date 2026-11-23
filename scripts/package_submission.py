"""Collect the verified course deliverables and check them after extraction."""

import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SUBMISSION = ARTIFACTS / "submission"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    artifact = ARTIFACTS / "leafcheck177-v1"
    freeze = read(artifact / "freeze.json")
    archive_check = read(ROOT / "docs/archive_verification.json")
    browser_check = read(ROOT / "docs/demo_verification.json")
    suites = read(ROOT / "docs/test_results.json")
    review = read(ROOT / "docs/submission_quality.json")
    restoration = read(ROOT / "docs/dataset_restoration_verification.json")
    demo = ARTIFACTS / "LeafCheck-final-demo.zip"
    checkpoint = freeze["checkpoint_sha256"]
    assert all(record["checkpoint_sha256"] == checkpoint for record in (archive_check, browser_check, suites))
    assert digest(demo) == archive_check["archive_sha256"]
    assert all(result["passed"] for result in suites["suites"]) and suites["review_notebook"]["executed_without_errors"]
    assert restoration["frozen_fingerprint"] == freeze["dataset_fingerprint"] and restoration["all_fingerprint_files_match"]
    for name in ("report", "deck"):
        assert digest(ROOT / review[name]["path"]) == review[name]["sha256"]
    notebook = ROOT / suites["review_notebook"]["executed_copy"]
    assert digest(notebook) == suites["review_notebook"]["executed_sha256"]
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(["git", "archive", "--format=tar", source_revision], cwd=ROOT)
    model_dir = "source/artifacts/leafcheck177-v1"
    freeze_entry = f"{model_dir}/freeze.json"
    output = ARTIFACTS / "LeafCheck-CSC427-submission.zip"
    temporary = output.with_suffix(".tmp")
    manifest = {}
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as package:
        def include(name, content):
            package.writestr(name, content)
            manifest[name] = hashlib.sha256(content).hexdigest()

        with tarfile.open(fileobj=io.BytesIO(source)) as tree:
            for member in tree:
                if member.isfile():
                    include(f"source/{member.name}", tree.extractfile(member).read())
        release = ROOT / "data/processed/leafcheck177_final"
        locked_release = read(ROOT / "data/manifests/leafcheck177_frozen.json")
        assert locked_release["fingerprint"] == freeze["dataset_fingerprint"]
        assert all(digest(release / name) == expected for name, expected in locked_release["files"].items())
        for name in ("README.md", "dataset_info.json", "exclusions.csv", *locked_release["files"]):
            include(f"source/data/release_metadata/leafcheck177_final/{name}", (release / name).read_bytes())
        for path in sorted((release / "metadata").rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                include(f"source/data/release_metadata/leafcheck177_final/{path.relative_to(release).as_posix()}", path.read_bytes())
        for name in ("LeafCheck_Report.docx", "LeafCheck_Defense.pptx"):
            include(name, (SUBMISSION / name).read_bytes())
        include("LeafCheck-final-demo.zip", demo.read_bytes())
        include("source/cnn_final_model_review.executed.ipynb", notebook.read_bytes())
        for suite in suites["suites"]:
            log = (ROOT / suite["log"]).resolve()
            assert log.is_relative_to(ROOT / "runs/final177/verification")
            assert digest(log) == suite["log_sha256"]
            include(f"source/{log.relative_to(ROOT).as_posix()}", log.read_bytes())
        for example in browser_check["examples"]:
            screenshot = (ROOT / example["screenshot"]).resolve()
            assert screenshot.is_relative_to(ROOT / "runs/final177/verification")
            assert digest(screenshot) == example["screenshot_sha256"]
            include(f"source/{screenshot.relative_to(ROOT).as_posix()}", screenshot.read_bytes())
        for name, expected in read(artifact / "artifact.json")["files"].items():
            path = (artifact / name).resolve()
            assert path.is_relative_to(artifact.resolve()) and digest(path) == expected
            include(f"{model_dir}/{name}", path.read_bytes())
        include(f"{model_dir}/artifact.json", (artifact / "artifact.json").read_bytes())
        for run in sorted((ROOT / "runs/final177").iterdir()):
            status_path = run / "status.json"
            if status_path.is_file() and read(status_path).get("state") == "complete":
                for name in ("history.csv", "status.json", "config.json", "environment.json", "lineage.json", "validation_metrics.json"):
                    if (run / name).exists():
                        include(f"source/runs/final177/{run.name}/{name}", (run / name).read_bytes())
        selected_run = Path(freeze["run"])
        if (selected_run / "last.pt").is_file():
            include(f"source/runs/final177/{selected_run.name}/last.pt", (selected_run / "last.pt").read_bytes())
        handoff = f"""# LeafCheck CSC-427 submission

Read LeafCheck_Report.docx for the report and LeafCheck_Defense.pptx for the
defense deck. The executed notebook at source/cnn_final_model_review.executed.ipynb
contains measured outputs. Its editable source is source/cnn_final_model_review.ipynb.
Run either from the extracted source folder with the documented research environment.

For the demo, extract LeafCheck-final-demo.zip and run Launch LeafCheck.command
inside LeafCheck. The app README covers setup on another Mac, including its
local Python dependencies. The separate demo archive passed a fresh extraction
and launcher check. The app needs its packaged weights and no training dataset.

Source revision: {source_revision}
Selected checkpoint SHA-256: {checkpoint}
Dataset fingerprint: {freeze['dataset_fingerprint']}

source/README.md and source/data/README.md give training and restoration commands.
Exact training reproduction requires a compatible environment and the original
frozen release or exact master files. The frozen membership and audit records
under source/data/release_metadata allow restoration from the original master.
The exported notebooks and report read the
project's documented artifact layout. Saved evaluation results and application
samples are included. Re-running the evaluation requires the original data.
"""
        include("START_HERE.md", handoff.encode())
        package.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    with tempfile.TemporaryDirectory(prefix="leafcheck-submission-") as folder:
        destination = Path(folder).resolve()
        with zipfile.ZipFile(temporary) as package:
            names = package.namelist()
            assert set(names) == {*manifest, "MANIFEST.json"}
            assert len(names) == len(set(names))
            assert all((destination / name).resolve().is_relative_to(destination) for name in names)
            package.extractall(destination)
        assert all(digest(destination / name) == expected for name, expected in manifest.items())
        assert read(destination / freeze_entry)["checkpoint_sha256"] == checkpoint
        assert digest(destination / model_dir / "model.pt") == checkpoint
        assert digest(destination / "LeafCheck-final-demo.zip") == archive_check["archive_sha256"]
        assert not (destination / "source/data/processed").exists()
        # Check the default artifact path from the extracted source using one
        # packaged photo. The full test split stays outside the archive.
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("LEAFCHECK_REVIEW_ARTIFACT", None)
        bundle = destination / model_dir
        example = read(bundle / "demo_examples.json")[0]
        standalone = subprocess.check_output([
            sys.executable, "-m", "leafcheck.inference", "--artifact", "artifacts/leafcheck177-v1",
            str(bundle / example["sample_file"])], cwd=destination / "source", env=environment, text=True)
        prediction = json.loads(standalone)
        assert prediction["label"] == example["predicted_class"]
        assert abs(prediction["score"] - example["score"]) < 1e-4
    temporary.replace(output)
    record = {"archive": str(output.relative_to(ROOT)), "bytes": output.stat().st_size,
              "sha256": digest(output), "source_revision": source_revision,
              "checkpoint_sha256": checkpoint, "files_verified_from_fresh_extraction": len(manifest),
              "fresh_extraction_standalone_prediction": {"label":prediction["label"], "score":prediction["score"]}}
    (SUBMISSION / "verification.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
