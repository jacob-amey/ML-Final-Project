"""Install an explicitly selected evaluated bundle; never choose a newest run."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""): result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(); source = args.artifact.resolve()
    manifest = json.loads((source / "artifact.json").read_text())
    if not manifest.get("final_evaluation_complete"):
        raise ValueError("Complete the frozen final evaluation before installing a release")
    if manifest.get("external_protocol_sha256") and not manifest.get("supplementary_evaluation_complete"):
        raise ValueError("Complete the registered supplementary evaluation before installing this release")
    models = Path(__file__).resolve().parents[1] / "models"; models.mkdir(exist_ok=True)
    relative = Path("final177") / manifest["checkpoint_sha256"][:16]
    destination = models / relative; destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        temporary = Path(tempfile.mkdtemp(prefix=".final177-", dir=models))
        try:
            for name, expected in manifest["files"].items():
                src = (source / name).resolve()
                if not src.is_relative_to(source): raise ValueError("Invalid artifact member path")
                if digest(src) != expected: raise ValueError(f"Artifact fingerprint mismatch: {name}")
                target = temporary / name; target.parent.mkdir(exist_ok=True, parents=True); shutil.copyfile(src, target)
            shutil.copyfile(source / "artifact.json", temporary / "artifact.json")
            temporary.rename(destination)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
    for name, expected in manifest["files"].items():
        if digest(destination / name) != expected: raise ValueError(f"Installed fingerprint mismatch: {name}")
    # Copy only the two explicitly selected demo photos into the public web tree.
    examples = []
    for record in json.loads((source / "demo_examples.json").read_text()):
        suffix = Path(record["sample_file"]).suffix
        name = "final_" + record["purpose"]
        asset = models.parent / "web/assets" / (name + suffix)
        temporary = asset.with_suffix(asset.suffix + ".tmp")
        shutil.copyfile(source / record["sample_file"], temporary); temporary.replace(asset)
        examples.append({"id": name, "name": "Held-out " + record["purpose"], "image": "assets/" + asset.name,
                         "actual_class": record["actual_class"], "description": "Frozen test example, selected after evaluation",
                         "source_sha256": digest(asset)})
    example_path = models.parent / "web/assets/final_examples.json"
    temporary = example_path.with_suffix(".tmp"); temporary.write_text(json.dumps(examples, indent=2)); temporary.replace(example_path)
    reference = models / "final_model.json"; temporary = reference.with_suffix(".tmp")
    temporary.write_text(json.dumps({"artifact": relative.as_posix(), "model_version": manifest["model_version"],
                                    "checkpoint_sha256": manifest["checkpoint_sha256"]}, indent=2)); temporary.replace(reference)
    print("Installed final model:", destination)
    print("Restart LeafCheck to load this explicit reference.")


if __name__ == "__main__": main()
