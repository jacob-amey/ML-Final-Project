"""Compare completed validation-only seed repeats without opening the test set."""

import argparse
import json
from pathlib import Path
from .common import atomic_json, sha256, write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs=2,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    records=[]
    for run in args.runs:
        status=json.loads((run / "status.json").read_text())
        assert status["state"]=="complete" and status["checkpoint_reload_verified"]
        config=json.loads((run / "config.json").read_text())
        metrics=json.loads((run / "validation_metrics.json").read_text())
        assert abs(metrics["macro_f1"]-status["best"]["macro_f1"])<1e-12
        assert metrics["sample_count"]==16836 and len(metrics["per_class"])==177
        records.append({"run":run.name,"seed":config["seed"],"config":config,"metrics":metrics,
                        "selected_epoch":status["best"]["epoch"],"checkpoint_sha256":sha256(run / "best_model.pt")})
    left,right=records
    descriptive={"seed","experiment","hypothesis","reference"}
    assert {k:v for k,v in left["config"].items() if k not in descriptive}=={k:v for k,v in right["config"].items() if k not in descriptive}
    assert left["seed"]!=right["seed"]
    common_files=["dataset.json","class_to_idx.json","preprocessing.json","class_weights.json"]
    core_files=["model.py","train.py","preprocessing.py","data.py"]
    assert all(sha256(args.runs[0]/name)==sha256(args.runs[1]/name) for name in common_files)
    assert all(sha256(args.runs[0]/"source/leafcheck"/name)==sha256(args.runs[1]/"source/leafcheck"/name) for name in core_files)
    rows=[]
    for a,b in zip(left["metrics"]["per_class"],right["metrics"]["per_class"]):
        assert (a["class_index"],a["class_name"],a["support"])==(b["class_index"],b["class_name"],b["support"])
        rows.append({"class_name":a["class_name"],"validation_images":a["support"],
                     f"f1_seed{left['seed']}":a["f1"],f"f1_seed{right['seed']}":b["f1"],
                     "absolute_f1_difference":abs(a["f1"]-b["f1"]),
                     f"recall_seed{left['seed']}":a["recall"],f"recall_seed{right['seed']}":b["recall"],
                     "zero_observed_recall_both":a["recall"]==0 and b["recall"]==0})
    rows.sort(key=lambda row:(-row["absolute_f1_difference"],row["class_name"]))
    write_csv(args.output.with_suffix(".csv"),rows)
    scores=[r["metrics"]["macro_f1"] for r in records]
    summary={"evaluation_split":"validation only","input_size":left["config"]["input_size"],
             "controlled_config":{k:v for k,v in left["config"].items() if k not in descriptive},
             "matching_numeric_configs_and_core_training_sources_verified":True,
             "runs":[{k:r[k] for k in ["run","seed","selected_epoch","checkpoint_sha256"]} | {"macro_f1":r["metrics"]["macro_f1"],"accuracy":r["metrics"]["accuracy"]} for r in records],
             "mean_macro_f1":sum(scores)/2,"macro_f1_range":max(scores)-min(scores),
             "classes_with_f1_difference_at_least_point2":sum(row["absolute_f1_difference"]>=.2 for row in rows),
             "classes_with_zero_recall_in_both":sum(row["zero_observed_recall_both"] for row in rows),
             "largest_class_differences":rows[:10],
             "limitations":"Two selected-checkpoint seed results are limited stability evidence, not significance or a confidence interval. Very small validation supports can produce large per-class swings. No test predictions enter this comparison."}
    atomic_json(args.output.with_suffix(".json"),summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="largest_class_differences"},indent=2))


if __name__=="__main__":main()
