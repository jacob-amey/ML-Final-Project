"""Recompute the frozen study's scores from saved records without image data."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from math import isclose
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = Path("runs/final177/inference_comparison/residual177_224_seed434-d4_probability_mean-predictions.csv")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def indexed(records):
    result = {}
    for row in records:
        image_id = row["image_id"]
        assert image_id not in result, "Repeated image ID"
        result[image_id] = row
    return result


def compare_scores(actual, saved):
    for key in ("correct", "sample_count", "accuracy", "macro_f1"):
        assert isclose(actual[key], saved[key], rel_tol=0, abs_tol=1e-12), (key, actual[key], saved[key])


def metrics(matrix, labels):
    actual = [sum(row) for row in matrix]
    predicted = [sum(column) for column in zip(*matrix)]
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    total = sum(actual)
    per_class = {}
    for i in labels:
        tp = matrix[i][i]
        denom = actual[i] + predicted[i]
        per_class[i] = {
            "support": actual[i], "precision": tp / predicted[i] if predicted[i] else 0,
            "recall": tp / actual[i] if actual[i] else 0, "f1": 2 * tp / denom if denom else 0,
        }
    return {
        "correct": correct, "sample_count": total, "accuracy": correct / total,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(per_class),
    }, per_class


def tally(expected, predictions, mapping, rule=None):
    if predictions is not None:
        assert predictions.keys() == expected.keys(), "Wrong prediction membership"
    matrix = [[0 for _ in mapping] for _ in mapping]
    groups = [set() for _ in mapping]
    for image_id, row in expected.items():
        actual = mapping[row["class_name"]]
        if predictions is None:
            predicted = mapping[rule]
        else:
            answer = predictions[image_id]
            predicted = mapping[answer["predicted_class"]]
            assert answer["actual_class"] == row["class_name"]
            assert answer["group_id"] == row["group_id"]
            assert answer["image_path"] == row.get("output_relative_path", row.get("image_path"))
            if "actual_index" in answer:
                assert int(answer["actual_index"]) == actual and int(answer["predicted_index"]) == predicted
            assert (answer["correct"].lower() == "true") == (actual == predicted)
        matrix[actual][predicted] += 1
        groups[actual].add(row["group_id"])
    return matrix, groups


def check_details(per_class, saved_rows, mapping, groups):
    assert [mapping[row["class_name"]] for row in saved_rows] == list(per_class)
    for row in saved_rows:
        idx = mapping[row["class_name"]]
        for key, expected in per_class[idx].items():
            assert isclose(float(row[key]), expected, rel_tol=0, abs_tol=1e-12), (row["class_name"], key)
        if row.get("group_support") is not None:
            assert int(row["group_support"]) == len(groups[idx])


def numeric_matrix(path):
    return [[int(cell) for cell in line.split(",")] for line in path.read_text().splitlines()]


def verify(artifact, release):
    freeze = read(artifact / "freeze.json")
    manifest = read(artifact / "artifact.json")
    flags = ("final_evaluation_complete", "reference_evaluation_complete", "supplementary_evaluation_complete")
    assert all(manifest[key] is True for key in flags)
    assert all(digest(artifact / name) == expected for name, expected in manifest["files"].items())
    for key, name in {
        "checkpoint_sha256": "model.pt", "config_sha256": "config.json",
        "preprocessing_sha256": "preprocessing.json", "class_mapping_sha256": "class_to_idx.json",
        "external_protocol_sha256": "external_protocol.json", "external_manifest_sha256": "external_manifest.csv",
    }.items():
        assert digest(artifact / name) == freeze[key], name
    reference = freeze["comparison_reference"]
    assert digest(artifact / "reference/model.pt") == reference["checkpoint_sha256"]
    assert manifest["checkpoint_sha256"] == freeze["checkpoint_sha256"]
    assert manifest["inference_policy"] == freeze["inference_policy"] == {"name": "d4_probability_mean"}
    selection = read(ROOT / "docs/final_model_selection.json")
    assert selection["test_predictions_used"] is False
    assert selection["confidence_threshold"] is None and selection["calibration"] == "none"
    assert selection["checkpoint_sha256"] == freeze["checkpoint_sha256"]
    assert selection["selected_policy"] == freeze["inference_policy"]
    assert selection["deployment_validation"] == freeze["deployment_validation"]
    assert selection["reference_checkpoint_sha256"] == reference["checkpoint_sha256"]
    for key in ("preprocessing.json", "reference/preprocessing.json"):
        p = read(artifact / key)
        assert p["statistics_split"] == "train" and p["statistics_images"] == 59008

    locked = read(ROOT / "data/manifests/leafcheck177_frozen.json")
    hashes = {name: digest(release / name) for name in locked["files"]}
    assert hashes == locked["files"]
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    assert fingerprint == locked["fingerprint"] == freeze["dataset_fingerprint"] == reference["dataset_fingerprint"]
    mapping = read(artifact / "class_to_idx.json")
    assert mapping == read(release / "class_to_idx.json") == read(artifact / "reference/class_to_idx.json")
    assert mapping == {r["class_name"]: int(r["index"]) for r in rows(artifact / "class_order.csv")}
    assert list(mapping.values()) == list(range(177)) and list(mapping) == locked["classes"]

    included = indexed(r for r in rows(release / "manifest.csv") if r["status"] == "included")
    splits = {split: {} for split in ("train", "val", "test")}
    group_split = {}
    for image_id, row in included.items():
        split = row["split"]
        assert mapping[row["class_name"]] == int(row["class_id"])
        assert row["group_id"] and group_split.setdefault(row["group_id"], split) == split
        if split != "train":
            assert row["previously_used"] == row["training_only"] == "False"
        splits[split][image_id] = row
    assert [len(split) for split in splits.values()] == [59008, 16836, 8425]
    assert all(len({row["class_name"] for row in split.values()}) == 177 for split in splits.values())

    result = {}
    test = splits["test"]
    train_counts = Counter(row["class_name"] for row in splits["train"].values())
    rule = max(mapping, key=train_counts.get)
    baseline = read(artifact / "baseline_metrics.json")
    assert baseline["rule"]["class_name"] == rule and baseline["rule"]["training_count"] == train_counts[rule]
    for name, directory, weight in (
        ("primary", artifact, freeze["checkpoint_sha256"]),
        ("compact177", artifact / "reference", reference["checkpoint_sha256"]),
        ("frequency_rule", artifact, None),
    ):
        prediction = indexed(rows(directory / "test_predictions.csv")) if weight else None
        saved = read(directory / "test_metrics.json") if weight else baseline
        assert saved["dataset_fingerprint"] == fingerprint
        if weight:
            assert saved["checkpoint_sha256"] == weight
        matrix, groups = tally(test, prediction, mapping, rule)
        result[name], per_class = metrics(matrix, range(177))
        compare_scores(result[name], saved)
        assert matrix == saved["confusion_matrix"]
        check_details(per_class, saved["per_class"], mapping, groups)
        if weight:
            check_details(per_class, rows(directory / "per_class_metrics.csv"), mapping, groups)
        if name == "primary":
            assert matrix == numeric_matrix(artifact / "confusion_matrix.csv")
            assert sum(row["support"] < 10 for row in per_class.values()) == saved["classes_with_fewer_than_10_test_images"] == 69
            assert prediction is not None
            for example in read(artifact / "demo_examples.json"):
                answer = prediction[example["image_id"]]
                for key in ("actual_class", "predicted_class", "image_path"):
                    assert answer[key] == example[key]
                assert float(answer["score"]) == example["score"]
                assert digest(artifact / example["sample_file"]) == example["image_id"].removeprefix("sha256:")

    validation, _ = tally(splits["val"], indexed(rows(ROOT / VALIDATION)), mapping)
    result["validation"], _ = metrics(validation, range(177))
    compare_scores(result["validation"], freeze["deployment_validation"])

    protocol = read(artifact / "external_protocol.json")
    external = indexed(r for r in rows(artifact / "external_manifest.csv") if r["status"] == "included")
    assert len(external) == protocol["included_images"] == 183
    assert not external.keys() & included.keys()
    support = Counter(row["class_name"] for row in external.values())
    assert support == protocol["class_counts"]
    labels = [mapping[name] for name in sorted(support)]
    assert len(labels) == 3
    external_rule = max(support, key=train_counts.get)
    assert external_rule == protocol["conditional_rule"]["class_name"]
    assert train_counts[external_rule] == protocol["conditional_rule"]["training_count"]
    matrix, groups = tally(external, indexed(rows(artifact / "external_predictions.csv")), mapping)
    result["supplement"], per_class = metrics(matrix, labels)
    saved = read(artifact / "external_metrics.json")
    assert saved["checkpoint_sha256"] == freeze["checkpoint_sha256"]
    assert saved["inference_policy"] == freeze["inference_policy"]["name"]
    compare_scores(result["supplement"], saved)
    check_details(per_class, saved["per_class"], mapping, groups)
    check_details(per_class, rows(artifact / "external_per_class.csv"), mapping, groups)
    assert matrix == saved["confusion_matrix"]
    collapsed = [[matrix[i][j] for j in labels] + [sum(matrix[i][j] for j in range(177) if j not in labels)]
                 for i in labels]
    assert collapsed == numeric_matrix(artifact / "external_confusion_matrix.csv")
    constant, _ = tally(external, None, mapping, external_rule)
    result["supplement_rule"], _ = metrics(constant, labels)
    compare_scores(result["supplement_rule"], saved["conditional_rule"])

    latency = read(artifact / "latency.json")
    assert latency["checkpoint_bytes"] == (artifact / "model.pt").stat().st_size
    assert len(latency["measurements_ms"]) == latency["timed_calls"] and 0 < latency["p50_ms"] <= latency["p95_ms"]
    return {"checkpoint_sha256": freeze["checkpoint_sha256"], "reference_sha256": reference["checkpoint_sha256"],
            "release_fingerprint": fingerprint, "bundle_files_verified": len(manifest["files"]),
            "groups_verified": len(group_split), "frequency_rule_label": rule, "supplement_rule_label": external_rule,
            "metrics": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts/leafcheck177-v1")
    parser.add_argument("--release", type=Path, default=ROOT / "data/release_metadata/leafcheck177_final")
    args = parser.parse_args()
    print(json.dumps(verify(args.artifact, args.release), indent=2))


if __name__ == "__main__":
    main()
