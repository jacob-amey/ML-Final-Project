"""Execute the predeclared validation-only scaling screens, safely resumable.

Never opens the test split or packages/selects a final artifact. A completed
screen earns a second seed only at the prospectively documented gain threshold.
"""
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from leafcheck.common import atomic_json, utc_now

child = None
stopping = False


def stop(signum, frame):
    global stopping
    stopping = True
    if child is not None and child.poll() is None:
        child.send_signal(signal.SIGINT)


def command(args, log):
    global child
    if stopping:
        raise SystemExit("Stopped; completed work and resumable checkpoint retained")
    with log.open("a") as output:
        child = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        code = child.wait()
    if code or stopping:
        raise SystemExit(f"Child stopped ({code}); inspect {log}; rerun this driver to resume")


def run(config_path, run_path):
    status_path = run_path / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if (run_path / "config.json").exists():
        assert json.loads((run_path / "config.json").read_text()) == json.loads(config_path.read_text()), "Run/config mismatch; do not silently resume different settings"
    if status.get("state") != "complete":
        args = ["-m", "leafcheck.train", "--run", str(run_path)]
        if (run_path / "last.pt").exists():
            args += ["--resume"]
        else:
            assert not run_path.exists(), "Unexpected incomplete directory; investigate rather than overwrite"
            args += ["--config", str(config_path)]
        command(args, ROOT / "runs/final177" / f"{run_path.name}_training.log")
    status = json.loads(status_path.read_text())
    assert status["state"] == "complete" and status["checkpoint_reload_verified"]
    command(["-m", "leafcheck.validation_analysis", str(run_path)], run_path / "analysis.log")
    command(["-m", "leafcheck.report", str(run_path)], run_path / "analysis.log")
    return status


def main():
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    reference = json.loads((ROOT / "docs/seed_stability_160_color.json").read_text())
    baseline = next(r["macro_f1"] for r in reference["runs"] if r["seed"] == 434)
    threshold = max(.005, reference["macro_f1_range"])
    results = []
    for tag in ["wide", "224"]:
        config_path = ROOT / f"configs/residual177_{tag}_seed434.json"
        run_path = ROOT / f"runs/final177/residual177_{tag}_seed434"
        atomic_json(ROOT / "docs/active_training.json", {"state":"scaling_screen", "run":str(run_path.relative_to(ROOT)), "updated_utc":utc_now(), "test_predictions_inspected":False})
        status = run(config_path, run_path)
        gain = status["best"]["macro_f1"] - baseline
        result = {"direction":tag, "first_seed":status, "reference_seed434_macro_f1":baseline,
                  "screen_gain":gain, "repeat_trigger":threshold, "repeat_required":gain >= threshold}
        if result["repeat_required"]:
            repeat_config = config_path.with_name(f"residual177_{tag}_seed435.json")
            config = json.loads(config_path.read_text()); config["seed"] = 435
            if repeat_config.exists():
                assert json.loads(repeat_config.read_text()) == config
            else:
                atomic_json(repeat_config, config)
            repeat_path = run_path.with_name(f"residual177_{tag}_seed435")
            atomic_json(ROOT / "docs/active_training.json", {"state":"scaling_seed_repeat", "run":str(repeat_path.relative_to(ROOT)), "updated_utc":utc_now(), "test_predictions_inspected":False})
            result["second_seed"] = run(repeat_config, repeat_path)
            command(["-m","leafcheck.seed_analysis","--runs",str(run_path),str(repeat_path),"--output",str(ROOT/f"docs/seed_stability_{tag}")], ROOT/f"runs/final177/{tag}_seed_analysis.log")
            pair = json.loads((ROOT/f"docs/seed_stability_{tag}.json").read_text())
            result["paired_mean_gain"] = pair["mean_macro_f1"] - reference["mean_macro_f1"]
            result["accepted_by_declared_rule"] = result["paired_mean_gain"] >= .005
        else:
            result["accepted_by_declared_rule"] = False
            result["limitation"] = "One-seed screen below declared repeat trigger; not proof scaling cannot help."
        results.append(result)
        atomic_json(ROOT/"docs/scaling_comparison.json", {"updated_utc":utc_now(),"evaluation_split":"validation only","reference":"docs/seed_stability_160_color.json","results":results,"test_predictions_used":False})
        command(["-m","leafcheck.experiments"], ROOT/"runs/final177/scaling_experiments.log")
    print("Declared scaling procedures completed; inspect results before final model freeze.", flush=True)


if __name__ == "__main__":
    main()
