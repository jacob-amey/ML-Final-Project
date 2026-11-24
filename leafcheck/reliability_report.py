"""Summarize the completed reliability study and produce class-level audit files."""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import DATA, ROOT, atomic_json, sha256, write_csv
from .reliability_study import ece
from .upload_study import ARTIFACT, read, rows


def class_audit(out, run):
    membership = [r for r in rows(DATA / "manifest.csv") if r["status"] == "included"]
    count = Counter((r["split"], r["class_name"]) for r in membership)
    groups = {}
    for row in membership:
        groups.setdefault((row["split"], row["class_name"]), set()).add(row["group_id"])
    results = {}
    for split, metrics in (("val", read(run / "baseline.json")["master"]),
                           ("test", read(ARTIFACT / "test_metrics.json"))):
        table = []
        total = metrics["sample_count"]
        matrix = np.array(metrics["confusion_matrix"])
        for report in metrics["per_class"]:
            i = report["class_index"]
            name = report["class_name"]
            support = report["support"]
            assert support == count[split, name]
            tp = int(matrix[i, i])
            fp = int(matrix[:, i].sum()) - tp
            fn = support - tp
            tn = total - tp - fp - fn
            table.append({**report, "train_files": count["train", name],
                          "train_groups": len(groups["train", name]), "val_files": count["val", name],
                          "test_files": count["test", name], "split_groups": len(groups[split, name]),
                          "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                          "false_positive_rate": fp / (fp + tn), "false_negative_rate": fn / support,
                          "one_vs_rest_accuracy": (tp + tn) / total})
        write_csv(out / f"per_class_{split}.csv", table)
        results[split] = {"support": total, "macro_f1": metrics["macro_f1"], "top1_accuracy": metrics["accuracy"],
                          "macro_precision": float(np.mean([r["precision"] for r in table])),
                          "balanced_accuracy_macro_recall": float(np.mean([r["recall"] for r in table])),
                          "weighted_f1": sum(r["f1"] * r["support"] for r in table) / total,
                          "zero_f1_classes": sum(r["f1"] == 0 for r in table),
                          "min_f1": min(r["f1"] for r in table)}
    return results


def plots(run, out):
    field = read(run / "field_metrics.json")
    labels = [r["label"] for r in field["per_class_screen"]]
    full = field["full_score_metrics"]
    index = {r["class_name"]: r["class_index"] for r in full["per_class"]}
    matrix = np.array(full["confusion_matrix"])
    indices = [index[label] for label in labels]
    focused = matrix[np.ix_(indices, indices)]
    focused = np.column_stack([focused, matrix[indices].sum(1) - focused.sum(1)])
    readable = [label.split("__")[1].replace("_", " ").title() for label in labels]
    fig, ax = plt.subplots(figsize=(7.8, 4), layout="constrained")
    img = ax.imshow(focused, cmap="Blues")
    ax.set_xticks(range(4), readable + ["Other label"])
    ax.set_yticks(range(3), readable)
    ax.set(xlabel="CNN top label before screening", ylabel="Publisher reference",
           title="Potato field check, 218 images and 177 eligible outputs")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, str(focused[i, j]), ha="center", va="center",
                    color="white" if focused[i, j] > focused.max() * .55 else "#132136")
    fig.colorbar(img, ax=ax, shrink=.8, label="Photos")
    fig.savefig(out / "field_confusions.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 3.7), layout="constrained")
    left = np.zeros(2)
    a, b = field["accept_all"], field["screen"]
    for name, values, color, ink in (
        ("Correct suggestion", [a["correct_and_accepted"], b["correct_and_accepted"]], "#187481", "white"),
        ("Incorrect suggestion", [a["incorrect_and_accepted"], b["incorrect_and_accepted"]], "#96384b", "white"),
        ("Withheld", [0, field["accept_all"]["support"] - b["accepted"]], "#cbd3dc", "#132136"),
    ):
        ax.barh(["Original app", "With uncertainty screen"], values, left=left, label=name, color=color)
        for i, value in enumerate(values):
            if value >= 20:
                ax.text(left[i] + value / 2, i, str(value), color=ink, ha="center", va="center")
        left += values
    ax.invert_yaxis()
    ax.set(xlabel="Photos", title="The rule reduces wrong suggestions by withholding more results")
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(.5, -.25), frameon=False)
    fig.savefig(out / "field_suggestions.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    out = run / "report"
    out.mkdir(exist_ok=True)
    leaderboard = []
    for path in sorted(run.glob("*_seed*/status.json")):
        state = read(path)
        assert state["state"] == "complete" and state["reload_verified"]
        best = state["best"]
        checkpoint = path.parent / "head.pt"
        assert state["sha256"] == sha256(checkpoint)
        leaderboard.append({"recipe": state["identity"]["recipe"], "seed": state["identity"]["seed"],
                            "epochs": state["completed_epochs"], "selected_epoch": best["epoch"],
                            "training_count": state["identity"]["training_count"],
                            "checkpoint_sha256": state["sha256"], "config_sha256": state["identity"]["config_sha256"],
                            "master_accuracy": best["master"]["accuracy"], "master_macro_f1": best["master"]["macro_f1"],
                            "master_wrong_crop": best["master"]["wrong_crop"], "supplement_correct": best["supplement"]["correct"],
                            "supplement_macro_f1": best["supplement"]["macro_f1"],
                            "supplement_wrong_crop": best["supplement"]["wrong_crop"], "promoted": False})
    write_csv(out / "experiments.csv", leaderboard)
    results = class_audit(out, run)
    val = rows(run / "val.csv")
    p = np.load(run / "incumbent_validation_probabilities.npy")
    mask = np.array([r["master"] == "True" for r in val])
    y = np.array([int(r["class_id"]) for r in val])[mask]
    confidence = p[mask].max(1)
    correct = p[mask].argmax(1) == y
    results["val"].update({
        "top3_accuracy": float(np.any(np.argsort(p[mask], axis=1)[:, -3:] == y[:, None], 1).mean()),
        "ece15": ece(confidence, correct),
        "high_score_errors_at_least_0_9": int((~correct & (confidence >= .9)).sum()),
        "median_score_correct": float(np.median(confidence[correct])),
        "median_score_incorrect": float(np.median(confidence[~correct])),
    })
    atomic_json(out / "classification_audit.json", results)
    plots(run, out)
    print(results, flush=True)


if __name__ == "__main__":
    main()
