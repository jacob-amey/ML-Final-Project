"""Resume the registered rice sequence with one trainer per run directory."""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_scaling_comparison as workflow
from leafcheck.common import sha256
from leafcheck.evaluate_rice_field_study import PLAN, ARMS


def read(path):
    return json.loads(path.read_text())


def assert_no_writer(run):
    processes = subprocess.check_output(["ps", "-axo", "command"], text=True)
    for arm in ARMS:
        directory = run / arm
        for form in (directory, directory.relative_to(ROOT)):
            marker = f" --run {form}"
            if any(" -m leafcheck.train " in row and
                   (row.endswith(marker) or marker + " " in row) for row in processes.splitlines()):
                raise RuntimeError(f"Another trainer owns {directory}")
    for form in (run, run.relative_to(ROOT)):
        if any(" -m leafcheck.evaluate_rice_field_study " in row and f" --run {form} " in row
               for row in processes.splitlines()):
            raise RuntimeError(f"Another evaluator owns {run}")


def finish_arm(run, arm, config):
    directory = run / arm
    status = directory / "status.json"
    saved_config = directory / "config.json"
    if saved_config.is_file() and read(saved_config) != read(config):
        raise ValueError(f"Changed registered configuration for {arm}")
    prior = read(status) if status.is_file() else {}
    if prior.get("state") == "complete":
        if not prior["checkpoint_reload_verified"]:
            raise ValueError(f"Completed {arm} lacks its reload evidence")
        print(f"Reusing completed {arm}", flush=True)
        return
    if directory.exists():
        if not (directory / "last.pt").is_file():
            raise ValueError(f"Preserve incomplete {directory} and inspect it before recovery")
        mode = ["--resume"]
    else:
        mode = ["--config", str(config.relative_to(ROOT))]
    relative = str(directory.relative_to(ROOT))
    workflow.command(["-u", "-m", "leafcheck.train", "--run", relative, *mode], run / f"{arm}.log")
    trained = read(status)
    if trained.get("state") != "complete" or not trained["checkpoint_reload_verified"]:
        raise RuntimeError(f"{arm} is incomplete. Resume only after reviewing its checkpoint.")


def run_study(run):
    if not run.is_relative_to(ROOT):
        raise ValueError("Choose a local study run directory in the repository")
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".sequence.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_no_writer(run)
        plan = read(PLAN)
        result = run / "comparison/result.json"
        for arm in ARMS:
            finish_arm(run, arm, ROOT / plan["configs"][arm])
        if result.exists():
            if read(result)["plan_sha256"] != sha256(PLAN):
                raise ValueError("Existing comparison uses a different registered plan")
            print(f"Reusing completed comparison at {result}", flush=True)
            return
        relative = str(run.relative_to(ROOT))
        workflow.command(["-u", "-m", "leafcheck.evaluate_rice_field_study", "--run", relative,
                          "--output", relative + "/comparison"], run / "comparison.log")
        if not result.is_file():
            raise RuntimeError("Evaluation ended without its result receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGINT, workflow.stop)
    signal.signal(signal.SIGTERM, workflow.stop)
    run_study(args.run.resolve())
