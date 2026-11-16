"""Evaluate the already frozen full-data compact reference on the shared holdout."""

import csv
from collections import Counter
import json
import time
from .common import atomic_json, sha256, utc_now, write_csv
from .data import PixelCache, build_cache
from .inference import Predictor
from .train import evaluate


def evaluate_reference(directory, frozen, freeze):
    record=freeze["comparison_reference"]; reference=directory / record["directory"]
    assert record["dataset_fingerprint"]==frozen["fingerprint"]
    assert sha256(reference / "model.pt")==record["checkpoint_sha256"]
    metrics_path=reference / "test_metrics.json"
    if metrics_path.exists():
        metrics=json.loads(metrics_path.read_text())
        assert metrics["checkpoint_sha256"]==record["checkpoint_sha256"]
        predictions=list(csv.DictReader((reference / "test_predictions.csv").open()))
        for row in predictions:
            row["actual_index"]=int(row["actual_index"]);row["predicted_index"]=int(row["predicted_index"])
            row["correct"]=row["correct"]=="True"
    else:
        predictor=Predictor(reference / "model.pt","cpu")
        assert predictor.classes==frozen["classes"] and predictor.spec["statistics_split"]=="train"
        build_cache("test",predictor.spec["input_size"],frozen)
        data=PixelCache("test",predictor.spec["input_size"],frozen,allow_test=True)
        assert len(data)==8425
        start=time.perf_counter()
        metrics,predictions=evaluate(predictor.model,data,predictor.spec,predictor.device,predictor.classes,128,True)
        metrics.update({"model_name":record["name"],"checkpoint_sha256":record["checkpoint_sha256"],
                        "dataset_fingerprint":frozen["fingerprint"],"class_count":177,"selected_epoch":record["selected_epoch"],
                        "input_size":predictor.spec["input_size"],"evaluation_device":"cpu","evaluation_batch_size":128,
                        "evaluated_utc":utc_now(),"evaluation_seconds":time.perf_counter()-start,"comparison_protocol":record["protocol"]})
        write_csv(reference / "test_predictions.csv",predictions)
        atomic_json(metrics_path,metrics)
    tp=Counter(row["actual_index"] for row in predictions if row["correct"])
    actual=Counter(row["actual_index"] for row in predictions);predicted=Counter(row["predicted_index"] for row in predictions)
    f1=sum(2*tp[i]/(actual[i]+predicted[i]) if actual[i]+predicted[i] else 0 for i in range(177))/177
    assert len(predictions)==metrics["sample_count"]==8425
    assert sum(row["correct"] for row in predictions)==metrics["correct"]
    assert abs(f1-metrics["macro_f1"])<1e-12
    write_csv(reference / "per_class_metrics.csv",metrics["per_class"])
    print("Frozen compact reference:", {k:metrics[k] for k in ["correct","sample_count","accuracy","macro_f1"]},flush=True)
    return metrics
