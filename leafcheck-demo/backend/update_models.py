"""Refresh the earlier pilot/Create ML bundles; install the final CNN with bundle_final.py."""

import os
from pathlib import Path
import shutil

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent


def main():
    sources = []
    for run_folder in ("leafcheck_small", "leafcheck_starter300"):
        checkpoints = list((PROJECT_ROOT / "runs" / run_folder).glob("*/best_model.pt"))
        if not checkpoints:
            raise SystemExit(f"No saved checkpoint found in runs/{run_folder}. Run its training notebook first.")
        checkpoint = max(checkpoints, key=lambda path: path.parent.name)
        sources.append((checkpoint, APP_ROOT / "models" / run_folder))

    configured = os.environ.get("LEAFCHECK_COREML_MODEL")
    apple = Path(configured).expanduser() if configured else PROJECT_ROOT / "PlantDataML.mlproj/Models/PlantDataML 1.mlmodel"
    if not apple.is_file():
        raise SystemExit("An exported .mlmodel file is needed to bundle the Create ML classifier.")

    for checkpoint, destination in sources:
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("best_model.pt", "validation_metrics.json", "test_metrics.json"):
            source = checkpoint.parent / filename
            target = destination / filename
            if source.exists():
                temporary = target.with_suffix(target.suffix + ".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(target)
            elif target.exists():
                target.unlink()
        print(f"Updated {destination.name} from {checkpoint.parent.name}")

    destination = APP_ROOT / "models/createml/PlantDataML.mlmodel"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    shutil.copy2(apple, temporary)
    temporary.replace(destination)
    print("Updated Apple Create ML")
    print("The three earlier-model files are now inside the local app. Restart LeafCheck to use them.")


if __name__ == "__main__":
    main()
