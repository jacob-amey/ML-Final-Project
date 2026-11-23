"""Build the course Word report from the evaluated local model bundle."""

import argparse
import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
TITLE = "Plant Health Classification and an Interactive Application"


def read(path):
    return json.loads(Path(path).read_text())


def add_page_number(section):
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), " PAGE ")
    paragraph._p.append(field)


def document_settings():
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.left_margin = section.right_margin = section.top_margin = section.bottom_margin = Inches(1)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)
    style.font.color.rgb = RGBColor.from_string("171C19")
    style.paragraph_format.line_spacing = 1.15
    style.paragraph_format.space_after = Pt(7)
    for name, size in [("Title", 25), ("Heading 1", 17), ("Heading 2", 12)]:
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string("000000")
        style.paragraph_format.space_after = Pt(6)
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "CSC-427 final project report"
    add_page_number(section)
    return doc


def table(doc, headings, rows, widths):
    grid = doc.add_table(rows=1, cols=len(headings))
    grid.autofit = False
    for column, width in zip(grid.columns, widths):
        column.width = Inches(width)
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:color"), "D9D9D9")
        borders.append(border)
    grid._tbl.tblPr.append(borders)
    for index, values in enumerate([headings, *rows]):
        row = grid.rows[0] if index == 0 else grid.add_row()
        for cell, value, width in zip(row.cells, values, widths):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            shading = OxmlElement("w:shd")
            shading.set(qn("w:fill"), "DFE7E2" if index == 0 else "F6F8F7" if index % 2 == 0 else "FFFFFF")
            cell._tc.get_or_add_tcPr().append(shading)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(3)
            paragraph.paragraph_format.space_after = Pt(3)
            run = paragraph.add_run(str(value))
            run.bold = index == 0
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def figure(doc, path, caption, width=6.1):
    doc.add_picture(str(path), width=Inches(width))
    paragraph = doc.add_paragraph(caption)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.runs[0].italic = True


def heading(doc, title):
    doc.add_page_break()
    doc.add_heading(title, level=1)


def pretty(label):
    return label.replace("__", " / ").replace("_", " ")


def build(artifact, output):
    freeze = read(artifact / "freeze.json")
    manifest = read(artifact / "artifact.json")
    assert all(manifest[key] for key in ("final_evaluation_complete", "reference_evaluation_complete", "supplementary_evaluation_complete"))
    config = read(artifact / "config.json")
    state = read(artifact / "status.json")
    validation = freeze["deployment_validation"]
    test = read(artifact / "test_metrics.json")
    compact = read(artifact / "reference/test_metrics.json")
    rule = read(artifact / "baseline_metrics.json")
    external = read(artifact / "external_metrics.json")
    latency = read(artifact / "latency.json")
    policy = read(artifact / "inference_policy_selection.json")
    duration = read(ROOT / "docs/duration_trial_decision.json")
    app = read(ROOT / "docs/archive_verification.json")
    browser = read(ROOT / "docs/demo_verification.json")
    assert app["checkpoint_sha256"] == browser["checkpoint_sha256"] == freeze["checkpoint_sha256"]
    docs = ROOT / "docs"
    examples = read(artifact / "demo_examples.json")
    input_size = config["input_size"]
    policy_names = {"single_view": "One standard view", "d4_probability_mean": "Mean of eight orientation scores",
                    "train_prior_quarter": "Training-frequency logit adjustment"}
    policy_name = policy_names[freeze["inference_policy"]["name"]]
    doc = document_settings()
    doc.add_paragraph("CSC-427 Final Project", style="Subtitle")
    doc.add_paragraph(TITLE, style="Title")
    doc.add_paragraph("LeafCheck study report")
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph("A user uploads an image of a plant or affected tissue. LeafCheck returns one of 177 cataloged plant conditions and displays its scores alongside earlier classroom models. The target classes combine a crop or species with a healthy condition, disease, pest or other inherited collection label. A prediction counts as correct when the highest-scoring label matches the reference annotation.")
    doc.add_paragraph(f"On the reserved 177-class test, the selected CNN matched {test['correct']:,} of {test['sample_count']:,} labels ({test['accuracy']:.2%} accuracy, {test['macro_f1']:.4f} macro-F1). Macro-F1 averages each class’s precision and recall balance with equal weight. Both measures describe performance on this release.")
    doc.add_heading("Related Work", level=1)
    doc.add_paragraph("Mohanty et al. showed that performance on controlled PlantVillage photos could drop on images from other conditions [1]. PlantDoc supplies more varied scenes and motivates checking a separate photo collection [2]. Residual connections give the feature extractor shorter paths for gradients [3]. LeafCheck implements its own four-stage network using that established design idea.")
    doc.add_heading("Contributions", level=1)
    doc.add_paragraph("The project combines an audited image release, a custom CNN trained from random initialization, controlled validation experiments, an explicit model bundle, and a local browser demo. The model role covers data, training, evaluation and inference handoff. The partner’s assigned role is the frontend and application interface. Codex assisted with the audited release, model studies, evaluation, integration and submission preparation. The repository contribution log preserves the evidence for individual work.")

    heading(doc, "Dataset and Split Integrity")
    doc.add_paragraph("The prepared release retains 84,269 images and all 177 labels from the 90,574-image master. The audit excluded 6,305 files for documented issues including identical content, transformed duplicates, conflicting label groups and invalid inputs. Each included row has a stable content ID, label, source and split. Source paths and available leaf IDs support traceability. Physical specimen IDs are missing for many files.")
    table(doc, ["Split", "Images", "Purpose"], [("Training", "59,008", "Fit weights, class weights and RGB statistics"),
          ("Validation", "16,836", "Select training recipes and checkpoints"), ("Reserved test", "8,425", "Score the frozen model and references")], [1.2, .75, 4.55])
    doc.add_paragraph("Known or inferred related images stay in one split. The audit checks exact bytes, decoded image orientation variants, available leaf IDs and related-view candidates. A supplemental review inspected 792 remaining representatives. Candidate screening cannot establish independence for unidentified plants or undiscovered edits. Training support ranges from 4 to 5,116 images per class. Twenty-six classes have fewer than 30 training images.")
    figure(doc, docs / "figures/class_distribution.png", "Figure 1. Training images per class. The release keeps all labels, including those with sparse support.", 5.8)

    heading(doc, "Custom CNN and Training")
    doc.add_paragraph(f"The selected residual CNN accepts a three-channel {input_size} × {input_size} image and has {freeze['parameters']:,} trainable parameters. A stride-2 convolution extracts local features. Four residual blocks refine and downsample them, using two 3 × 3 convolutions per block. Each block adds a shortcut, with a 1 × 1 projection when spatial size or channel count changes.")
    with (docs / "selected_architecture.csv").open(newline="") as stream:
        shapes = list(csv.DictReader(stream))
    table(doc, ["Stage", "Output shape", "Trainable parameters"], [(r["stage"], r["output_shape"], f"{int(r['trainable_parameters']):,}") for r in shapes], [2.1, 2.4, 2.0])
    doc.add_paragraph("Global average pooling reduces each final feature map to one value. Dropout of 0.2 regularizes training, and the last linear layer produces 177 logits. PyTorch initializes the weights, computes gradients and supplies standard layers. The architecture and training recipe define how those components work together.")
    doc.add_paragraph("Images use the first frame, corrected camera orientation and RGB conversion. Bilinear resize and gray padding preserve aspect ratio. The same deterministic preprocessing standardizes every channel using training-only mean and standard deviation. Training adds reflections, quarter turns and mild brightness and contrast changes.")
    doc.add_paragraph(f"AdamW starts at {config['learning_rate']} with {config['weight_decay']} weight decay, batch size {config['batch_size']} and a validation-driven learning-rate schedule. Square-root inverse-frequency weights increase rare examples’ contribution to training cross-entropy. The selected run completed {state['completed_epochs']} full epochs and retained epoch {freeze['selected_epoch']}.")

    heading(doc, "Validation Experiments and Selection")
    doc.add_paragraph("All full runs used the same 59,008 training and 16,836 validation images. Each training checkpoint was selected by macro-F1, with lower unweighted validation cross-entropy breaking ties within 0.000001 and the earlier epoch retained for a remaining tie. Repeat seeds test sensitivity to initialization and training order.")
    rows = []
    for label, file in [("128 px, weighted", "128"), ("160 px, weighted", "160"),
                        ("160 px, illumination", "160_color"), ("160 px, stronger decay", "160_color_decay"),
                        ("224 px, illumination", "224"), ("256 px, illumination", "256"), ("256 px, longer budget", "256_long80")]:
        path = docs / f"seed_stability_{file}.json"
        if path.exists():
            pair = read(path)
            rows.append((label, *(f"{row['macro_f1']:.6f}" for row in pair["runs"]), f"{pair['mean_macro_f1']:.6f}"))
    table(doc, ["Recipe", "Seed 434", "Seed 435", "Mean F1"], rows, [3.05, 1.15, 1.15, 1.15])
    doc.add_paragraph("The full-data compact CNN reached 0.536531 validation macro-F1. Adding residual capacity raised it to 0.595327 in a matched seed. Increasing channel width later failed to improve the 160-pixel recipe. Moderate illumination variation improved both seeds, and the selected decay strength was 0.0001.")
    doc.add_paragraph(f"The paired 256-pixel change versus 224 was {duration['native_size_paired_gain']:+.6f} macro-F1. The declared promotion threshold was +0.005. The recorded duration decision and learning curves preserve any late training follow-up. Two seeds provide limited evidence about run variability.")
    doc.add_paragraph(f"The deployed scoring policy is {policy_name.lower()}. Validation also tested training-frequency logit adjustment [4] and mean scores across eight transformed views [5]. Their two-seed gains were " + ", ".join(f"{policy_names[row['policy']].lower()} {row['mean_gain']:+.6f}" for row in policy["policies"]) + f". The delivered checkpoint and policy achieved {validation['macro_f1']:.6f} macro-F1 and {validation['accuracy']:.2%} accuracy on validation.")
    figure(doc, artifact / "learning_curves.png", "Figure 2. The selected training run. The dashed marker identifies the epoch selected on validation.", 5.9)

    heading(doc, "Frozen Test Results")
    table(doc, ["Method", "Correct / total", "Accuracy", "Macro-F1"], [(name, f"{r['correct']:,} / {r['sample_count']:,}", f"{r['accuracy']:.2%}", f"{r['macro_f1']:.6f}")
          for name, r in [("Selected custom CNN", test), ("Compact177 full-data reference", compact), ("Most frequent training class", rule)]], [2.95, 1.35, 1.1, 1.1])
    doc.add_paragraph(f"The frequency rule always predicts {pretty(rule['rule']['class_name'])}. Each method uses the same 8,425 held-out images. The checkpoint, class order, preprocessing, score policy and fair reference were committed to the freeze record before evaluation. The two earlier CNN pilots predict three labels. Create ML covered 177 labels, but 5,832 new test images fall within its declared old training split, so its live outputs are exploratory.")
    no_recall = sum(row["recall"] == 0 for row in test["per_class"])
    doc.add_paragraph(f"The test contains {test['known_group_count']:,} known or inferred groups. Mean accuracy across groups is {test['group_balanced_accuracy']:.2%}. There are {test['classes_with_fewer_than_10_test_images']} classes with fewer than 10 test images and {no_recall} with zero observed recall. Per-class precision, recall, F1, support and every prediction accompany the bundle.")
    figure(doc, artifact / "major_confusions.png", "Figure 3. Largest directed confusions in the frozen 177-class test.", 6.0)

    heading(doc, "Supplementary Results and Demonstration")
    conditional = external["conditional_rule"]
    doc.add_paragraph(f"A second evaluation used 183 screened PlantDoc images representing three tomato labels. The CNN correctly classified {external['correct']} of 183 ({external['accuracy']:.2%}, three-label macro-F1 {external['macro_f1']:.4f}). The conditional frequency rule got {conditional['correct']} of 183 ({conditional['accuracy']:.2%}, macro-F1 {conditional['macro_f1']:.4f}). All 177 CNN outputs were eligible. Related sources appear elsewhere in the master, and the healthy proxy uses the source label Tomato leaf.")
    table(doc, ["Demo example", "Reference label", "CNN prediction", "Score"], [(r["purpose"].capitalize(), pretty(r["actual_class"]), pretty(r["predicted_class"]), f"{r['score']:.6f}") for r in examples], [1.15, 2.2, 2.25, .9])
    doc.add_paragraph("The browser sends JPEG or PNG bytes to the local Python server. The server checks format and size, converts the photo to orientation-corrected RGB and runs each available model. JavaScript renders the returned labels and scores. The final loader verifies bundle hashes and reads the selected checkpoint reference.")
    doc.add_paragraph(f"The frozen CNN checkpoint is {latency['checkpoint_bytes']:,} bytes. On the tested Mac, CPU inference including image preparation had median latency {latency['p50_ms']:.1f} ms and 95th percentile {latency['p95_ms']:.1f} ms over {latency['timed_calls']} timed calls. The fresh extracted application’s four-model HTTP requests had median {app['http_latency']['p50_ms']:.1f} ms. Browser and API checks covered actual photos and unreadable uploads.")
    doc.add_heading("Discussion and Next Step", level=2)
    doc.add_paragraph("The gap between accuracy and macro-F1 matters for rare conditions. Some inherited labels describe broad symptoms, and predictions can be sensitive to background and camera conditions. The score ranks supported categories and is uncalibrated. LeafCheck produces one label per photo and cannot establish a diagnosis or reliably identify unfamiliar inputs. Add independently collected, verified photos with specimen and session IDs for rare and poorly represented categories before making stronger field claims.")

    heading(doc, "Conclusion and Project Management")
    doc.add_paragraph("The project delivered a custom 177-class CNN, a reproducible selection protocol and a local upload application. Its primary test gains and weak supplementary accuracy define different limits on its use. The team’s model and application responsibilities and Codex-assisted work are documented in docs/contributions.md.")
    doc.add_heading("Reproduction and Appendix", level=2)
    doc.add_paragraph("Source and configuration history: https://github.com/jacob-amey/ML-Final-Project. The delivery archive records its local source revision. See README.md for exact environment, cache, training, resume, evaluation and launch commands. Open cnn_final_model_review.ipynb in the LeafCheck kernel for the architecture, saved tables and two demo predictions. Reproducing training requires the supplied frozen release or the exact master bytes and source records in data/README.md.")
    doc.add_paragraph("Dataset SHA-256: " + freeze["dataset_fingerprint"])
    doc.add_paragraph("Selected weights SHA-256: " + freeze["checkpoint_sha256"])
    doc.add_paragraph("The model package contains the selected weights, class mapping, preprocessing parameters, scoring policy, evaluation outputs and two demo photos. The application archive can be extracted and run without training data or notebooks. Model architecture, development records, experiment notes and the two contribution roles are documented in the source checkout.")
    doc.add_heading("References", level=2)
    for ref in [
        "[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, ‘Using deep learning for image-based plant disease detection,’ Front. Plant Sci., vol. 7, art. 1419, 2016. doi: 10.3389/fpls.2016.01419.",
        "[2] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, ‘PlantDoc: A dataset for visual plant disease detection,’ Proc. CoDS-COMAD, pp. 249–253, 2020. doi: 10.1145/3371158.3371196.",
        "[3] K. He, X. Zhang, S. Ren, and J. Sun, ‘Deep residual learning for image recognition,’ Proc. CVPR, pp. 770–778, 2016. doi: 10.1109/CVPR.2016.90.",
        "[4] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, ‘Long-tail learning via logit adjustment,’ Proc. ICLR, 2021. https://arxiv.org/abs/2007.07314.",
        "[5] D. Shanmugam, D. Blalock, G. Balakrishnan, and J. Guttag, ‘Better aggregation in test-time augmentation,’ Proc. ICCV, pp. 1214–1223, 2021.",
    ]:
        doc.add_paragraph(ref)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.artifact, arguments.output)
