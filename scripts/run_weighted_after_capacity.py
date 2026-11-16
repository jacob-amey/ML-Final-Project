"""Resume the declared imbalance experiment after the sole capacity job completes."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
status = ROOT / "runs/final177/residual177_seed434/status.json"
print("Waiting for the capacity run, then resuming the imbalance experiment.", flush=True)
while True:
    if status.exists():
        record = json.loads(status.read_text())
        if record["state"] == "complete": break
        if record["state"] == "interrupted":
            raise SystemExit("Capacity run is interrupted; inspect/resume it before advancing.")
    time.sleep(5)
target = ROOT / "runs/final177/residual177_weighted_seed434"
record = json.loads((target / "status.json").read_text())
if record["state"] == "complete": raise SystemExit("Imbalance experiment already completed.")
print("Capacity run completed. Resuming all-data imbalance training.", flush=True)
subprocess.run([sys.executable, "-m", "leafcheck.train", "--run", str(target), "--resume"], cwd=ROOT, check=True)
