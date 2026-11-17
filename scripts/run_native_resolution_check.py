"""Run the conditional, predeclared 256-pixel comparison after 224 finishes.

Only train/validation data are accessible through the invoked development tools.
No final artifact is selected or evaluated here.
"""
import json
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_scaling_comparison as workflow
from leafcheck.common import atomic_json, utc_now


def main():
    signal.signal(signal.SIGINT, workflow.stop)
    signal.signal(signal.SIGTERM, workflow.stop)
    comparison = json.loads((ROOT / "docs/scaling_comparison.json").read_text())
    direction = next(r for r in comparison["results"] if r["direction"] == "224")
    assert direction["accepted_by_declared_rule"], "256 check is conditional on accepted paired224 evidence"
    reference = json.loads((ROOT / "docs/seed_stability_224.json").read_text())
    for record in reference["runs"]:
        status = json.loads((ROOT / "runs/final177" / record["run"] / "status.json").read_text())
        assert status["state"] == "complete" and status["checkpoint_reload_verified"]
    config_path = ROOT / "configs/residual177_256_seed434.json"
    config = json.loads(config_path.read_text())
    reference_config = reference["controlled_config"]
    descriptive = {"experiment", "hypothesis", "reference", "seed", "input_size"}
    assert {k:v for k,v in config.items() if k not in descriptive} == {k:v for k,v in reference_config.items() if k not in descriptive}
    assert config["input_size"] == 256 and config["seed"] == 434
    log = ROOT / "runs/final177/native_resolution_sequence.log"
    benchmark = ROOT / "runs/final177/diagnostics/native_resolution_benchmark.json"
    benchmark_complete = benchmark.exists() and len(json.loads(benchmark.read_text()).get("benchmarks", [])) == 2
    if not benchmark_complete:
        workflow.command(["-m","leafcheck.benchmark","--configs",str(ROOT/"configs/residual177_224_seed434.json"),str(config_path),"--output",str(benchmark)], log)
    results = []
    for seed in [434, 435]:
        if seed == 435 and not result["repeat_required"]:
            break
        path = config_path.with_name(f"residual177_256_seed{seed}.json")
        cfg = {**config, "seed":seed}
        if path.exists():
            assert json.loads(path.read_text()) == cfg
        else:
            atomic_json(path, cfg)
        run_path = ROOT / f"runs/final177/residual177_256_seed{seed}"
        atomic_json(ROOT/"docs/active_training.json", {"state":"native_resolution_check", "run":str(run_path.relative_to(ROOT)), "updated_utc":utc_now(), "test_predictions_inspected":False})
        status = workflow.run(path, run_path)
        results.append(status)
        if seed == 434:
            baseline = next(r["macro_f1"] for r in reference["runs"] if r["seed"] == 434)
            gain = status["best"]["macro_f1"] - baseline
            threshold = max(.005, reference["macro_f1_range"])
            result = {"reference":"docs/seed_stability_224.json", "screen_gain":gain, "repeat_trigger":threshold,
                      "repeat_required":gain >= threshold, "accepted_by_declared_rule":False}
        else:
            workflow.command(["-m","leafcheck.seed_analysis","--runs",str(run_path.with_name("residual177_256_seed434")),str(run_path),"--output",str(ROOT/"docs/seed_stability_256")], log)
            pair = json.loads((ROOT/"docs/seed_stability_256.json").read_text())
            result["paired_mean_gain"] = pair["mean_macro_f1"] - reference["mean_macro_f1"]
            result["accepted_by_declared_rule"] = result["paired_mean_gain"] >= .005
        atomic_json(ROOT/"docs/scaling_256_comparison.json", {"state":"repeat_pending" if seed == 434 and result["repeat_required"] else "complete", **result, "completed_procedures":results, "updated_utc":utc_now(), "test_predictions_used":False})
    workflow.command(["-m","leafcheck.experiments"], log)
    print("Native resolution check completed; review validation evidence before final freeze.", flush=True)


if __name__ == "__main__":
    main()
