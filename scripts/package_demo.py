"""Archive the verified local app and model bundles without training data."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "leafcheck-demo"
sys.path.insert(0, str(DEMO / "backend"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda:file.read(1024 * 1024), b""):value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/LeafCheck-final-demo.zip")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    from final_model import FinalCNN
    final = FinalCNN()  # Refuses missing, inconsistent or unevaluated releases.
    reference = json.loads((DEMO / "models/final_model.json").read_text())
    files = [DEMO / name for name in ["Launch LeafCheck.command", "README.md", "requirements.txt", "requirements-lock.txt"]]
    for folder in ["backend", "web"]:
        files.extend(path for path in (DEMO / folder).rglob("*") if path.is_file()
                     and "__pycache__" not in path.parts and not path.name.startswith("."))
    for folder in ["leafcheck_small", "leafcheck_starter300", "createml", reference["artifact"]]:
        directory = DEMO / "models" / folder
        if not directory.is_dir(): raise FileNotFoundError(directory)
        files.extend(path for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    files.append(DEMO / "models/final_model.json")
    manifest = {path.relative_to(DEMO).as_posix():digest(path) for path in sorted(set(files))}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = args.receipt or (ROOT / "docs/demo_archive.json" if output == parser.get_default("output")
                              else output.with_suffix(".json"))
    temporary = output.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(set(files)):
            archive.write(path, "LeafCheck/" + path.relative_to(DEMO).as_posix())
        archive.writestr("LeafCheck/PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2))
    temporary.replace(output)
    record = {"archive":str(output.relative_to(ROOT) if output.is_relative_to(ROOT) else output),
              "sha256":digest(output), "bytes":output.stat().st_size,
              "model_checkpoint_sha256":final.metadata["checkpoint_sha256"], "files":len(manifest),
              "source_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT).decode().strip(),
              "contains_training_dataset":False,"contains_notebooks":False,
              "restore":"Unzip into a new folder, install the app requirements in a compatible Python environment, then run its launcher or backend/server.py."}
    receipt.write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__ == "__main__":main()
