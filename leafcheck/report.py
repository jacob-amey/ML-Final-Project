"""Report-ready figures generated from recorded metrics, never invented results."""

import csv
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from .common import DATA, atomic_json, write_csv


def learning_curves(run, output=None):
    run = Path(run); output = Path(output or run)
    with (run / "history.csv").open() as file: rows = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for axis, key, title in zip(axes, ["loss", "accuracy", "macro_f1"], ["Loss", "Accuracy", "Macro F1 across 177 classes"]):
        for prefix, label in [("train", "Training"), ("validation", "Validation")]:
            axis.plot(epochs, [float(row[f"{prefix}_{key}"]) for row in rows], label=label)
        axis.set(xlabel="Completed epoch", ylabel=title, ylim=(0, None) if key == "loss" else (0, 1))
        axis.grid(alpha=.2); axis.legend()
    config = json.loads((run / "config.json").read_text())
    fig.suptitle(config["experiment"] + " — training loss includes configured class weights")
    fig.tight_layout(); fig.savefig(output / "learning_curves.png", dpi=170); plt.close(fig)


def evaluation_figures(directory, metrics, predictions, classes):
    directory = Path(directory)
    matrix = np.asarray(metrics["confusion_matrix"])
    np.savetxt(directory / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    write_csv(directory / "per_class_metrics.csv", metrics["per_class"])
    write_csv(directory / "class_order.csv", [{"index": i, "class_name": name} for i, name in enumerate(classes)])
    fig, axis = plt.subplots(figsize=(13, 11))
    normalized = np.divide(matrix, matrix.sum(1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(1, keepdims=True) != 0)
    heatmap = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    axis.set(title="Frozen test confusion matrix — row-normalized; labels in class_order.csv",
             xlabel="Predicted class index", ylabel="Actual class index")
    axis.set_xticks(range(0, len(classes), 10)); axis.set_yticks(range(0, len(classes), 10))
    fig.colorbar(heatmap, ax=axis, label="Fraction of this true class")
    fig.tight_layout(); fig.savefig(directory / "confusion_matrix.png", dpi=180); plt.close(fig)
    confusion = [{"actual_class": classes[i], "predicted_class": classes[j], "count": int(matrix[i, j]),
                  "actual_support": int(matrix[i].sum()), "fraction_of_actual_class": float(matrix[i, j] / matrix[i].sum())}
                 for i in range(len(classes)) for j in range(len(classes)) if i != j and matrix[i, j]]
    confusion.sort(key=lambda row: (-row["count"], row["actual_class"], row["predicted_class"]))
    write_csv(directory / "major_confusions.csv", confusion, fields=["actual_class", "predicted_class", "count", "actual_support", "fraction_of_actual_class"])
    top = list(reversed(confusion[:12]))
    fig, axis = plt.subplots(figsize=(12, 7))
    axis.barh([row["actual_class"] + " → " + row["predicted_class"] for row in top], [row["count"] for row in top], color="#42735d")
    axis.set(xlabel="Incorrect test images", title="Largest directed test confusions by image count", xlim=(0, None))
    fig.tight_layout(); fig.savefig(directory / "major_confusions.png", dpi=160); plt.close(fig)
    correct = sorted([row for row in predictions if row["correct"]], key=lambda row: row["image_id"])
    wrong = sorted([row for row in predictions if not row["correct"]], key=lambda row: -row["score"])
    chosen = []; used = set()
    for pool in [correct, wrong]:
        selected = []
        for row in pool:
            if row["actual_class"] not in used:
                selected.append(row); used.add(row["actual_class"])
            if len(selected) == 3: break
        chosen.extend(selected)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for axis, row in zip(axes.flat, chosen):
        with Image.open(DATA / row["image_path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB"); image.thumbnail((600, 500))
            axis.imshow(image)
        axis.set_title(f"True: {row['actual_class']}\nPred: {row['predicted_class']}\nSoftmax score: {row['score']:.3f}", fontsize=9)
        axis.axis("off")
    fig.suptitle("Frozen test examples — successes and confidently incorrect predictions")
    fig.tight_layout(); fig.savefig(directory / "test_examples.jpg", dpi=160); plt.close(fig)
    atomic_json(directory / "example_records.json", chosen)
    demo = directory / "samples"; demo.mkdir(exist_ok=True)
    # Preserve source bytes, including EXIF, for genuine upload/inference checks.
    import shutil
    success = next((row for row in correct if row["actual_class"] == "tomato__healthy"), correct[0])
    mistake = wrong[0] if wrong else None
    sample_records = []
    for name, row in [("success", success), ("mistake", mistake)]:
        if row:
            suffix = Path(row["image_path"]).suffix
            target = demo / (name + suffix); shutil.copyfile(DATA / row["image_path"], target)
            sample_records.append({**row, "sample_file": str(target.relative_to(directory)), "purpose": name,
                                   "note": "Chosen after final evaluation for demonstration; never used to tune the model."})
    atomic_json(directory / "demo_examples.json", sample_records)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("run", type=Path); args = parser.parse_args()
    learning_curves(args.run)
