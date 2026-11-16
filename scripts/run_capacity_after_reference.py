"""Sequential experiment queue; keep the sole accelerator run monitored."""

import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
status = ROOT / "runs/final177/plain177_seed434/status.json"
print("Waiting for the full-data reference to finish before the capacity experiment.", flush=True)
while True:
    if status.exists():
        record = json.loads(status.read_text())
        if record["state"] == "complete": break
        if record["state"] == "interrupted":
            raise SystemExit("Reference is interrupted; resume it before advancing the experiment queue.")
    time.sleep(5)
target = ROOT / "runs/final177/residual177_seed434"
if target.exists(): raise SystemExit("Capacity run already exists; inspect/resume it explicitly.")
print("Reference completed. Starting the controlled capacity experiment.", flush=True)
subprocess.run([sys.executable, "-m", "leafcheck.train", "--config", "configs/residual177.json",
                "--run", str(target)], cwd=ROOT, check=True)
