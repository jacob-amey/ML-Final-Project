"""Build submission source material from completed, frozen experiment records."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import ROOT, sha256, write_csv


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    directory = parser.parse_args().artifact
    artifact = read(directory / "artifact.json")
    assert artifact["final_evaluation_complete"] and artifact["supplementary_evaluation_complete"]
    assert artifact["reference_evaluation_complete"]
    assert sha256(directory / "model.pt") == artifact["checkpoint_sha256"]
    test = read(directory / "test_metrics.json"); rule = read(directory / "baseline_metrics.json")
    reference = read(directory / "reference/test_metrics.json")
    external = read(directory / "external_metrics.json"); latency = read(directory / "latency.json")
    config = read(directory / "config.json"); status = read(directory / "status.json")
    selected = read(directory / "freeze.json"); validation = read(directory / "validation_metrics.json")
    deployment_validation = selected.get("deployment_validation", validation)
    policy_names = {"single_view":"One standard image view", "d4_probability_mean":"Mean softmax scores over eight orientations",
                    "train_prior_quarter":"Training frequency adjustment by subtracting 0.25 log class prior from each logit"}
    policy_name = selected.get("inference_policy", {"name":"single_view"})["name"]
    policy_record = directory / "inference_policy_selection.json"
    policy_paragraph = ""
    if policy_record.exists():
        policy_results = read(policy_record)
        assert policy_results["selected_policy"]["name"] == policy_name
        effects = " and ".join(f"{policy_names[r['policy']]} ({r['mean_gain']:+.6f})" for r in policy_results["policies"])
        policy_paragraph = f"\nThe paired validation trial measured macro-F1 changes of {effects}. Each used the same trained CNN weights and validation images as the single-view control. The selected policy was included in the freeze.\n"
    experiments = list(csv.DictReader((ROOT / "docs/experiments.csv").open()))
    assert all(row["state"] == "complete" for row in experiments)
    # A zero tensor verifies software shapes only; it is never accuracy evidence.
    import torch
    from .inference import Predictor
    torch.set_num_threads(4)
    network = Predictor(directory, "cpu").model
    size = config["input_size"]
    x = torch.zeros(1, 3, size, size, dtype=torch.float32)
    stages = [{"stage":"Input", "output_shape":f"B×3×{size}×{size}",
               "trainable_parameters":0, "purpose":"RGB image after deterministic preprocessing"}]
    with torch.inference_mode():
        for index, block in enumerate(network.features):
            x = block(x)
            stages.append({"stage":"Stem" if index == 0 else f"Residual block {index}",
                           "output_shape":"B×"+"×".join(str(d) for d in x.shape[1:]),
                           "trainable_parameters":sum(p.numel() for p in block.parameters() if p.requires_grad),
                           "purpose":"Learn local features and downsample" if index != 1 else "Refine features at the same spatial size"})
        for module in network.classifier:
            x = module(x)
            purposes = {"AdaptiveAvgPool2d":"Average each feature channel across space", "Flatten":"Convert pooled channels to a vector",
                        "Dropout":"Training regularizer; inactive during evaluation", "Linear":"Produce one logit per supported class"}
            kind = type(module).__name__
            stages.append({"stage":kind, "output_shape":"B×"+"×".join(str(d) for d in x.shape[1:]),
                           "trainable_parameters":sum(p.numel() for p in module.parameters() if p.requires_grad),
                           "purpose":purposes[kind]})
    assert tuple(x.shape) == (1, 177)
    assert sum(row["trainable_parameters"] for row in stages) == selected["parameters"]
    write_csv(ROOT / "docs/selected_architecture.csv", stages)
    architecture_table = "\n".join(f"| {r['stage']} | {r['output_shape']} | {r['trainable_parameters']:,} | {r['purpose']} |" for r in stages)
    figures = ROOT / "docs/figures"; figures.mkdir(exist_ok=True)
    for name in ["learning_curves.png", "confusion_matrix.png", "major_confusions.png", "external_confusion_matrix.png"]:
        shutil.copyfile(directory / name, figures / name)
    fresh_runs = [row for row in experiments if not row["parent_run"] or row.get("additional_training_seconds")]
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6.2, len(fresh_runs) * .48)))
    for row in fresh_runs:
        history = list(csv.DictReader((ROOT / "runs/final177" / row["run"] / "history.csv").open()))
        axes[0].plot([int(r["epoch"]) for r in history], [float(r["validation_macro_f1"]) for r in history], label=row["run"], linestyle="--" if row["parent_run"] else "-")
    axes[0].set(xlabel="Completed full-data epoch", ylabel="Validation macro-F1 (177 classes)", ylim=(0, 1))
    axes[0].grid(alpha=.2); axes[0].legend(fontsize=7)
    names = [row["run"].replace("177", "").replace("_seed", "\nseed ") for row in fresh_runs]
    selected_name = Path(selected["run"]).name
    axes[1].barh(range(len(names)), [float(row["validation_macro_f1"]) for row in fresh_runs],
                 color=["#bd812c" if row["run"] == selected_name else "#527a67" for row in fresh_runs])
    axes[1].set_yticks(range(len(names)), names, fontsize=8)
    axes[1].set(xlabel="Selected-checkpoint validation macro-F1", xlim=(0, 1))
    axes[1].invert_yaxis()
    for i, row in enumerate(fresh_runs): axes[1].text(float(row["validation_macro_f1"])+.012, i, f"{float(row['validation_macro_f1']):.3f}", va="center", fontsize=8)
    fig.suptitle("Development evidence — shared frozen train/validation splits")
    fig.tight_layout(); fig.savefig(figures / "experiment_comparison.png", dpi=180); plt.close(fig)
    training_support = {r["class_name"]:int(r["training_images"]) for r in csv.DictReader((ROOT / "docs/training_class_support.csv").open())}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for axis, result, title in zip(axes, [validation, test], ["Selected validation checkpoint", "Frozen final test"]):
        for sparse, color, label in [(False,"#527a67","At least 10 evaluation images"),(True,"#bd812c","Fewer than 10 evaluation images")]:
            points = [r for r in result["per_class"] if (r["support"]<10)==sparse]
            axis.scatter([training_support[r["class_name"]] for r in points], [r["f1"] for r in points], color=color, label=label, alpha=.65, s=22)
        axis.set(xscale="log",xlabel="Training images for the label (log scale)",ylabel="Per-class F1",ylim=(-.03,1.03),title=title)
        axis.grid(alpha=.15); axis.legend(fontsize=7,loc="lower right")
    fig.suptitle("Class support and performance — descriptive, not a causal comparison")
    fig.tight_layout(); fig.savefig(figures / "support_and_f1.png", dpi=180); plt.close(fig)
    rows = "\n".join(f"| {r['run']} | {r['completed_epochs']} | {r['selected_epoch']} | {float(r['validation_accuracy']):.4f} | {float(r['validation_macro_f1']):.4f} | {float(r['validation_loss']):.4f} | {r['decision']} |" for r in experiments)
    seed_rows = []
    for path in sorted((ROOT / "docs").glob("seed_stability_*.json")):
        evidence = read(path)
        assert evidence["matching_numeric_configs_and_core_training_sources_verified"]
        cfg = read(ROOT / "runs/final177" / evidence["runs"][0]["run"] / "config.json")
        widths = "/".join(str(c) for c in cfg["model"]["channels"])
        seed_rows.append(f"| {cfg['input_size']} | {widths} | {cfg['augmentation']} | {cfg['weight_decay']} | {evidence['mean_macro_f1']:.6f} | {evidence['macro_f1_range']:.6f} |")
    seed_table = "\n".join(seed_rows)
    confusion = list(csv.DictReader((directory / "major_confusions.csv").open()))[:8]
    confusion_rows = "\n".join(f"| {r['actual_class']} | {r['predicted_class']} | {r['count']} / {r['actual_support']} | {float(r['fraction_of_actual_class']):.1%} |" for r in confusion)
    rare = [r for r in test["per_class"] if r["support"] < 10]
    no_recall = [r for r in test["per_class"] if r["recall"] == 0]
    ext_rows = "\n".join(f"| {r['class_name']} | {r['support']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |" for r in external["per_class"])
    conditional = external["conditional_rule"]
    augmentation_description = {
        "dihedral": "whole-image reflections and quarter turns",
        "dihedral_color": "whole-image reflections and quarter turns, plus independently sampled brightness and contrast factors from 0.9 to 1.1",
    }[config["augmentation"]]
    channel_description = "/".join(str(c) for c in config["model"]["channels"])
    report = f'''# Plant Health Classification and an Interactive Application

Generated from completed saved records. This is report source material, not a
claim that a formatted 5–10 page report or presentation has been submitted.

## Introduction and related work

LeafCheck classifies a photograph into one of 177 inherited plant-condition
categories. Labels combine crop/species with conditions, including healthy
tissue, disease, pests and some non-leaf plant parts. It is single-label image
classification. The project pairs a randomly initialized custom CNN with an
existing local comparison application. PlantVillage research motivates caution
about controlled-background accuracy versus new photographs [1]. PlantDoc
provides more varied scenes [2]. Residual connections [3] motivate the capacity
experiment; the project does not claim to invent this established technique.

## Contribution and scope

The implemented contribution is an audited 177-label release, shared train/inference
preprocessing, a reproducible custom architecture comparison, validation-led
selection, sealed final evaluation, explicit artifact packaging and a tested
four-model comparison workflow. The older three-tomato-label pilots are preserved
as historical experiments. The earlier MobileNet proposal is superseded; no
pretrained weights are used by the final CNN. Codex-assisted implementation and
execution are disclosed in `contributions.md`; the student must understand and
review the work and follow course policy.

## Dataset and leakage precautions

The release contains 84,269 images across all 177 labels: 59,008 training,
16,836 validation and 8,425 test. The original 90,574-image master is intact.
The audit records 6,305 exclusions with provenance and reasons. Related-image
groups, duplicates and documented prior exposure informed split assignment.
Content IDs provide traceability, not physical plant identity. Training support
ranges from 4 to 5,116 images per label (median 100); 26 labels have fewer than
30 training images. No new downloaded images or generated evaluation images
were added during this final execution. See `final_dataset_audit.md`,
`dataset_summary.json`, `training_class_support.csv` and pinned source manifests.
Source reuse terms must be reviewed before redistribution of the full dataset.

Dataset fingerprint: `{selected['dataset_fingerprint']}`.

![Training distribution](figures/class_distribution.png)

## Methods and preprocessing

The selected model is `{config['model']['architecture']}`, seed {config['seed']},
with {selected['parameters']:,} trainable parameters. The table below traces a
zero-input software check; shape/parameter derivations are in
`model_defense_guide.md`. It uses a stride-2 convolutional stem, four custom
two-convolution residual blocks ({channel_description} channels), batch normalization,
ReLU, global average pooling, dropout {config['model']['dropout']} and 177 logits.
Framework layers/autograd implement the mathematics; all weights start randomly.

| Stage | Output tensor shape | Trainable parameters | Purpose |
| --- | --- | ---: | --- |
{architecture_table}

Preprocessing selects frame zero, corrects EXIF orientation, converts to RGB,
bilinearly resizes while preserving aspect ratio and center-pads with gray 128
to {config['input_size']}×{config['input_size']}. Float32 pixels are divided by 255
and standardized with saved training-only mean/std. Training alone uses
{augmentation_description}. These avoid hue shifts and lesion-removing crops,
but upright field scenes need not be rotation invariant. Augmented views are
not independent samples. Evaluation and inference share one deterministic function.

Training uses AdamW, initial learning rate {config['learning_rate']}, batch size
{config['batch_size']}, weight decay {config['weight_decay']}, gradient clipping
{config['gradient_clip']}, class-weight power {config['class_weight_power']}, and
cross-entropy on raw logits. Class weights use training counts only, are clipped
and normalized. Unweighted validation loss is reported separately from weighted
training loss. ReduceLROnPlateau follows validation macro-F1. The budget is
{config['max_epochs']} full epochs, minimum {config['min_epochs']}, early-stop
patience {config['patience']}; label smoothing {config.get('label_smoothing', 0.0)}. The selected procedure completed
{status['completed_epochs']} epochs; checkpoint epoch {selected['selected_epoch']}.

All fresh full-data runs visit all 59,008 training images per epoch. Configuration,
code/source fingerprints, versions, RNG/optimizer/scheduler states and atomic
best/resume checkpoints are retained. Training uses Apple M2 Max MPS, 30 GPU
cores, 32 GB unified RAM. The benchmark and tiny-learning/resume checks are under
`runs/final177/diagnostics`; source tests verify CPU and MPS interrupted resumes.

## Controlled experiments and selection

Checkpoint/configuration selection uses validation macro-F1 over all 177 labels,
then lower unweighted validation loss within 1e-6, then earlier epoch; effectively
tied configurations favor the smaller/faster model. Methods share frozen splits.
Architecture comparisons do not guarantee identical random streams. Seed repeats
investigate stability, not statistical significance. BatchNorm refinement reuses
a parent procedure; its cost must not be counted as another fresh 48-epoch run.

| Run | Epochs | Selected epoch | Val accuracy | Val macro-F1 | Val loss | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{rows}

| Input size | Channels | Training augmentation | Weight decay | Two-seed mean macro-F1 | Observed seed range |
| ---: | --- | --- | ---: | ---: | ---: |
{seed_table}

These are two selected validation checkpoints per recipe, not confidence
intervals or evidence of significance. Rare-class sampling variability and
repeated use of the validation set still limit the comparison.

![Development comparison](figures/experiment_comparison.png)

**Frozen selection rationale and stopping decision:** {selected['rationale']}

The selected checkpoint scored {validation['macro_f1']:.6f} macro-F1 and
{validation['accuracy']:.4%} accuracy with the original training validation
protocol. The delivered prediction policy is {policy_names[policy_name].lower()}.
With this policy, the selected checkpoint scored
{deployment_validation['macro_f1']:.6f} validation macro-F1 and
{deployment_validation['accuracy']:.4%} validation accuracy. Repeated selection
can favor matches to the validation split. The test seal and freeze records
preserve the separate holdout protocol.
{policy_paragraph}

![Selected learning curves](figures/learning_curves.png)

## Frozen final testing

| Method on the same 177-class holdout | Correct / total | Accuracy | Macro-F1 over 177 |
| --- | ---: | ---: | ---: |
| Selected custom CNN | {test['correct']} / {test['sample_count']} | {test['accuracy']:.4%} | {test['macro_f1']:.6f} |
| Compact177 full-data reference | {reference['correct']} / {reference['sample_count']} | {reference['accuracy']:.4%} | {reference['macro_f1']:.6f} |
| Most frequent training label: {rule['rule']['class_name']} | {rule['correct']} / {rule['sample_count']} | {rule['accuracy']:.4%} | {rule['macro_f1']:.6f} |

The compact reference is a separate 253,257-parameter, randomly initialized CNN,
trained on all 59,008 images and all 177 labels. Its validation-selected epoch
{reference['selected_epoch']} and checkpoint were frozen before any test results.
It is distinct from the earlier three-class Small CNN pilot. This held-out
comparison evaluates complete recipes; it does not isolate the causal effect of
architecture, weighting, duration or resolution. A better test result would not
be used to switch the already selected deliverable.

The test has {test['known_group_count']} known/inferred related-image groups.
Mean within-group correct fraction, averaged equally across groups, is
{test['group_balanced_accuracy']:.4%} (rule {rule['group_balanced_accuracy']:.4%}).
This is a sensitivity summary, not proof of independent biological sampling.
{len(rare)} labels have fewer than 10 test images; {len(no_recall)} labels have zero
observed test recall. Per-class precision, recall, F1, image/group support and
all predictions are retained in the bundle. A single rare-class success/failure
is unstable evidence; high overall accuracy does not imply reliable rare labels.

![Class support and performance](figures/support_and_f1.png)
![Frozen confusion matrix](figures/confusion_matrix.png)
![Largest directed confusions](figures/major_confusions.png)

| Actual | Predicted | Errors / actual support | Fraction of actual class |
| --- | --- | ---: | ---: |
{confusion_rows}

`test_examples.jpg`, `example_records.json` and `demo_examples.json` preserve
successes and confidently wrong predictions. Demo examples were chosen after
evaluation for explanation and were not used to tune the model.

## Supplementary three-label PlantDoc protocol

The preregistered supplementary collection retains 183 photos after 57 exact or
related overlaps were removed. Outputs still compete across all 177 labels.
The CNN gets {external['correct']}/183 correct ({external['accuracy']:.4%}), with
macro-F1 over the three represented tomato labels {external['macro_f1']:.6f}.
The training-frequency conditional tomato rule gets {conditional['correct']}/183
({conditional['accuracy']:.4%}), macro-F1 {conditional['macro_f1']:.6f}.
These macro scores have a different label denominator from the 177-class test.

| Reference label | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
{ext_rows}

![Supplementary confusion matrix](figures/external_confusion_matrix.png)

PlantDoc appears elsewhere in the master, so this is not an unseen-source test.
Generic `Tomato leaf` is an inherited healthy proxy rather than an independently
verified diagnosis. A small three-label collection does not validate other crops,
unfamiliar diseases or deployment in the field. Missing/cropped related views may
remain despite screening. Final test findings do not trigger more tuning here.

## Comparison integrity and application handoff

The pilots cover three labels and use different splits. The actual bundled
Create ML model covers all 177, but its declared earlier training data contains
5,832 new-test images, with 1,683 more in its old validation split. Its exposure
cannot support a fair new-test score. The website retains live exploratory
predictions without presenting the historical models as equivalent held-out
competitors. See `comparison_protocol.json`.

The final bundle pairs weights, architecture/runtime, mapping, preprocessing,
versions, freeze record and saved metrics. The website uses an explicit checkpoint
reference, CPU evaluation mode and a shared prediction contract. The uncalibrated
scores rank the supported labels. A high score can accompany an incorrect
prediction. The classifier
does not reliably reject unknown/non-plant images, localize symptoms, resolve
multiple diseases or provide a definitive diagnosis.

Checkpoint: `{selected['checkpoint_sha256']}`.
Checkpoint size: {latency['checkpoint_bytes']:,} bytes.
CPU batch-one inference on {latency['hardware']['cpu']} with {latency['torch_threads']} PyTorch threads: median
{latency['p50_ms']:.3f} ms, p95 {latency['p95_ms']:.3f} ms over 48 timed calls after eight warmups,
cycling four predetermined native test images. This includes decode, EXIF/RGB,
padding, normalization, network and softmax with warm local file cache; it excludes
HTTP and other comparison models. Power source at measurement:
{latency['hardware'].get('power_source','not recorded')}. Warm-filesystem model loading is
{latency['model_loading_ms_warm_filesystem']:.3f} ms. Training throughput and app-wide
request latency are different measurements. See `latency.json` and the separately
recorded demo verification/packaging evidence.

## Discussion and future work

The evidence supports this model only under the documented splits and inherited
labels. Missing specimen IDs, residual near-duplicate risks, coarse/ambiguous
condition labels, controlled backgrounds, imbalance and limited field imagery
constrain the conclusions. The strong accuracy/macro-F1 gap warrants caution.
Prioritize verified, traceable independent examples for weak/rare labels and
plant/session IDs, plus an independently collected field evaluation. Resolution,
augmentation or capacity changes would require new validation experiments and a
genuinely new independent final evaluation protocol after this test has been seen.
Compression/quantization require parity and performance checks if deployment
measurements justify them. No globally optimal model or target accuracy is claimed.

## Conclusion and project management

The selected artifact completes the documented custom 177-label training and
frozen evaluation procedure. Application verification is reported separately
in `demo_verification.json`; do not infer a browser check from offline metrics.
Use the exact README commands for training, interruption/resume, packaging,
inference and website launch. Machine-readable metrics, confusion matrices,
predictions and raw checkpoints remain recoverable local artifacts outside Git.

The user owns the data/model/evaluation role; the partner's assigned role is the
application/frontend. Codex-assisted execution is recorded explicitly rather
than attributed as independent student activity. Names, meeting participation,
team approval and institutional AI policy must be supplied/verified by the team.
The submission archive includes a formatted report and defense deck. The team
reviews contributions and course policy before formally submitting and presenting.

## References

[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for
image-based plant disease detection,” *Front. Plant Sci.*, vol. 7, art. 1419, 2016,
doi: [10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419).

[2] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra,
“PlantDoc: A dataset for visual plant disease detection,”
in *Proc. CoDS-COMAD*, 2020, pp. 249–253,
doi: [10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196).

[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in *Proc. CVPR*, 2016, pp. 770–778,
doi: [10.1109/CVPR.2016.90](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).

Dataset revisions, annotations and source links are in `data/README.md` and
`final_dataset_audit.md`; original code references include PyTorch documentation
and the architecture references above. No original pretrained implementation
or weights were copied into the final model.

## Appendix and project source

The configured project repository is
[jacob-amey/ML-Final-Project](https://github.com/jacob-amey/ML-Final-Project).
Final execution work is recorded on the local `codex/final-leafcheck177` branch;
the team must ensure the approved source revision is available to the instructor.
Large datasets/checkpoints follow repository storage rules. Artifact fingerprints,
local archive locations and restoration instructions accompany the delivered
model; a source checkout alone does not include the image collection or weights.
Dataset source links and pinned revisions are in `data/README.md`; the provided
master remains necessary for exact training-data restoration where no public
download is established. Source code, configurations and the review notebook
are identified in the project README. Framework API references include the
[PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html).
The numbered reference format follows the assignment's
[University of Pittsburgh IEEE guide](https://pitt.libguides.com/citationhelp/ieee).
'''
    (ROOT / "docs/final_model_report.md").write_text(report)
    print("Saved report source material and figures from completed frozen evidence.")


if __name__ == "__main__": main()
