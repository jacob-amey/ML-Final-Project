"""Build the course Word report from the evaluated local model bundle."""

import argparse
import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE
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
    for name, size in [("Title", 25), ("Subtitle", 11), ("Heading 1", 17), ("Heading 2", 12)]:
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string("000000")
        style.paragraph_format.space_after = Pt(6)
    title_style = doc.styles["Title"].element
    for border in title_style.xpath("./w:pPr/w:pBdr"):
        border.getparent().remove(border)
    doc.core_properties.title = TITLE
    doc.core_properties.author = "Jacob Amey and Demetre"
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


def add_link(paragraph, label, url):
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
    run = paragraph.add_run(label)
    run.font.color.rgb = RGBColor.from_string("17523E")
    run.underline = True
    link.append(run._r)
    paragraph._p.append(link)


def field_followup(doc, docs, freeze, include_site_audit=False):
    evidence = docs / "potato_context"
    okara = read(evidence / "okara_metrics.json")["methods"]
    tomato = read(evidence / "bangladesh_tomato_metrics.json")["methods"]
    validation = read(evidence / "validation.json")["automatic"]
    replay = read(evidence / "tom2024_matched.json")
    crops = read(evidence / "plantseg_regions_decision.json")
    demo = read(evidence / "demo_browser.json")
    decoding = read(docs / "field_optimization/crop_decoding.json")
    pakistan = read(docs / "field_optimization/pakistan_baseline.json")["results"]
    assert demo["checkpoint_sha256"] == freeze["checkpoint_sha256"]
    assert replay["base_checkpoint"] == crops["base_checkpoint"] == freeze["checkpoint_sha256"]
    assert not replay["test_opened"] and not crops["test_opened"]
    heading(doc, "Field Transfer and Upload Reliability")
    doc.add_paragraph("A follow-up evaluated unfamiliar image sources. The Okara potato collection [7] and the Bangladesh tomato collection [8] supplied publisher-labeled photos. Screening excluded potential duplicates and conflicting groups. Each collection stayed separate from fitting. The audited samples often showed isolated leaves, and undocumented specimen relationships can remain.")
    table(doc, ["Evaluation scope", "Correct / total", "Macro-F1"], [
        (name, f"{row['correct']:,} / {row['sample_count']:,}", f"{row['macro_f1']:.4f}")
        for name, row in [
            ("Okara automatic", okara["automatic_original"]),
            ("Okara with potato supplied", okara["masked_original"]),
            ("Bangladesh automatic", tomato["automatic_original"]),
            ("Bangladesh with tomato supplied", tomato["known_tomato_original"]),
        ]
    ], [3.5, 1.6, 1.4])
    row = tomato["automatic_original"]
    doc.add_paragraph(f"Automatic scoring named a different crop on {row['wrong_crop']} of {row['sample_count']} Bangladesh photos. Adjustment using a known crop addresses crop selection, while the remaining errors include confusion between early and late blight. Each source table averages only its represented classes. The potato option was verified through the browser and uses three conditional, uncalibrated scores. The tomato option was evaluated as a research diagnostic.")
    doc.add_paragraph(
        f"Eight-view averaging across two seeds improved master validation macro-F1 from "
        f"{validation['reference']['macro_f1']:.6f} to {validation['ensemble']['macro_f1']:.6f}. "
        f"The ensemble scored {tomato['automatic_two_seed']['correct']} of {row['sample_count']} "
        f"on Bangladesh automatic classification. New TOM2024 training photos [9] yielded "
        f"{replay['arms']['new_field']['master']['macro_f1']:.6f} master validation macro-F1 and "
        f"PlantSeg lesion crops [10] yielded {crops['arms']['new_field']['master']['macro_f1']:.6f}. "
        "Both training recipes missed their registered promotion criteria. The app continues to use the frozen checkpoint."
    )
    reward = decoding["decisions"]["crop_first"]
    doc.add_paragraph(
        f"Choosing the crop with the largest combined model score before ranking its conditions gained "
        f"{reward['field_f1_gain']:.4f} mean joint macro-F1 across the Sheikh tomato and Mduma potato "
        f"development slices. The gain fell below the registered threshold of 0.0200. This strengthens the "
        f"case for obtaining verification images from new capture groups before changing inference."
    )
    supplied = pakistan["reference_crop_diagnostic"]
    shown = pakistan["already_deployed_gate"]
    doc.add_paragraph(
        f"On a screened Pakistan development session [11], the CNN classified "
        f"{pakistan['correct']} of {pakistan['sample_count']} correctly, with joint macro-F1 "
        f"{pakistan['macro_f1']:.4f}. It named another crop on {pakistan['wrong_crop']} photos. "
        f"Supplying the source crop recovered {supplied['correct']} correct conditions, "
        f"and all {shown['accepted']} suggestions shown by the existing score screen were wrong. "
        "The session uses publisher visual labels and covers one capture day."
    )
    if include_site_audit:
        safari = read(docs / "field_optimization/imagesafari_baseline.json")
        automatic = safari["metrics"]["automatic_model"]
        framing = safari["framing_visual_review"]["by_framing"]
        foreground = framing["foreground_visible"]
        pilot = read(docs / "field_optimization/imagesafari_pilot.json")
        assert not pilot["passed_screen"] and not pilot["test_opened"]
        transfer = pilot["site_held_crop_decision"]["gain_over_control"]
        doc.add_paragraph(
            f"Image Safari [16] added site-held crop data. Automatic inference named the right crop "
            f"on {automatic['correct']} of {automatic['sample_count']} validation views and on "
            f"{foreground['automatic_crop_correct']} of {foreground['count']} views rated as having "
            "visible foreground plant detail. The framing audit was exploratory and the photos "
            "lack condition labels. A matched training pilot gained "
            f"{transfer:.4f} crop macro-F1 over its control and missed the 0.0300 gate."
        )


def tomato_transfer(doc, docs, freeze, include_crop_head=False):
    evidence = docs / "field_optimization"
    study = read(evidence / "tomato_matched.json")
    blend = read(evidence / "tomato_blend_development.json")
    holdout = read(evidence / "plantcity_confirmation.json")
    protocol = read(evidence / "plantcity_freeze.json")
    assert study["receipt"]["run"]["reference_checkpoint"] == blend["reference_checkpoint"] == freeze["checkpoint_sha256"]
    assert holdout["membership_and_metrics_independently_reconciled"] and not holdout["passed_confirmation"]
    assert protocol["created_utc"] < holdout["completed_utc"]
    heading(doc, "Field Transfer Experiments" if include_crop_head else "Tomato Transfer Experiment")
    training = study["training"]
    development = blend["selected_for_confirmation"]
    doc.add_paragraph(
        f"To test whether new field images improve condition recognition, two matched seeds trained "
        f"from random weights on {training['master']:,} master tomato images. Each epoch added "
        f"{training['field']:,} same-label master draws in the control or screened supplementary field "
        "images in the treatment. Controls stopped after 9 and 11 epochs. Both treatment runs reached "
        "the 12-epoch cap. Learning used ten tomato labels. Grouped Sheikh and Pakistan validation "
        "sources contributed equally to the development macro-F1."
    )
    table(doc, ["Tomato supplied in development", "Mean field macro-F1"], [
        ("Original CNN restricted to tomato", f"{study['conditional']['reference']['field']['macro_f1']:.4f}"),
        ("Master-only scratch controls", f"{study['summary']['control']['field_mean_macro_f1']:.4f}"),
        ("Field-augmented scratch models", f"{study['summary']['field']['field_mean_macro_f1']:.4f}"),
        ("50:50 original and field score blend", f"{development['mean_field_f1']:.4f}"),
    ], [4.4, 2.1])
    doc.add_paragraph(
        "The standalone specialists missed the master-validation guard. Routing the blend by the "
        "original model’s crop also reduced joint field macro-F1 by 0.0166 on both seeds. "
        "The 50:50 blend passed the development checks when tomato was supplied. Its selected "
        "weights and acceptance criteria were frozen before PlantCity inference."
    )
    table(doc, ["PlantCity [12] evaluation", "Correct / total", "Macro-F1"], [
        (label, f"{row['correct']} / {row['sample_count']}", f"{row['macro_f1']:.4f}")
        for label, row in [
            ("Original CNN, automatic", holdout["automatic_reference"]),
            ("Original CNN, tomato supplied", holdout["given_crop_reference"]),
            ("50:50 blend, tomato supplied", holdout["given_crop_blend"]),
        ]
    ], [3.9, 1.4, 1.2])
    automatic = holdout["automatic_reference"]
    doc.add_paragraph(
        f"PlantCity contributed screened originals covering 13 labels and six crops, including "
        f"six tomato conditions. Specimen IDs are unavailable. Automatic inference predicted a "
        f"different crop for {automatic['wrong_crop']} of {automatic['sample_count']} photos. "
        "The blend lost 0.0653 macro-F1 when tomato was supplied, so it stayed out of the app. "
        "These results motivate studying crop recognition separately on new capture groups."
    )
    if include_crop_head:
        crop = read(evidence / "crop_head_matched.json")
        check = read(evidence / "crop_head_confirmation.json")
        assert crop["both_seeds_passed_development"] and check["membership_and_metrics_independently_reconciled"]
        assert not check["passed_confirmation"]
        comparisons = check["results"].values()
        total = sum(item["reference"]["sample_count"] for item in comparisons)
        correct_before = sum(item["reference"]["correct"] for item in comparisons)
        correct_after = sum(item["candidate"]["correct"] for item in comparisons)
        center = read(evidence / "center_view.json")
        assert not center["passes_validation_screen"] and not center["test_opened"]
        doc.add_heading("Independent Crop Classifier", level=2)
        doc.add_paragraph(
            "A scratch CNN trained with 529 CCMT crop labels [13] reweighted crop scores. "
            "Both seeds improved grouped development macro-F1, mainly on potato. "
            f"On Holeta [14] and TLID [15], correct predictions rose from {correct_before} to "
            f"{correct_after} of {total}. Macro-F1 gained {check['mean_macro_f1_gain']:.4f} and "
            f"wrong crops fell {check['mean_crop_error_reduction']*100:.2f} points, below both gates. "
            f"A central-view blend changed field validation F1 by {center['field_gain']:+.4f}. "
            "The app retains the frozen CNN."
        )


def build(artifact, output, include_field_followup=False, include_tomato_transfer=False,
          include_crop_head=False, include_field_v8=False, include_field_v9=False):
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
    assert browser["extracted_app_check"]["archive_sha256"] == app["archive_sha256"]
    docs = ROOT / "docs"
    examples = read(artifact / "demo_examples.json")
    input_size = config["input_size"]
    policy_names = {"single_view": "One standard view", "d4_probability_mean": "Mean of eight orientation scores",
                    "train_prior_quarter": "Training-frequency logit adjustment"}
    policy_name = policy_names[freeze["inference_policy"]["name"]]
    doc = document_settings()
    doc.add_paragraph("CSC-427 Final Project", style="Subtitle")
    doc.add_paragraph(TITLE, style="Title")
    doc.add_paragraph("Jacob Amey and Demetre")
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph("A user uploads an image of a plant or affected tissue. LeafCheck returns one of 177 cataloged plant conditions and displays its scores alongside earlier classroom models. The target classes combine a crop or species with a healthy condition, disease, pest or other inherited collection label. A prediction counts as correct when the highest-scoring label matches the reference annotation.")
    doc.add_paragraph(f"On the reserved 177-class test, the selected CNN matched {test['correct']:,} of {test['sample_count']:,} labels ({test['accuracy']:.2%} accuracy, {test['macro_f1']:.4f} macro-F1). Macro-F1 averages each class’s precision and recall balance with equal weight. Both measures describe performance on this release.")
    doc.add_heading("Related Work", level=1)
    doc.add_paragraph("Mohanty et al. showed that performance on controlled PlantVillage photos could drop on images from other conditions [1]. PlantDoc supplies more varied scenes and motivates checking a separate photo collection [2]. Residual connections give the feature extractor shorter paths for gradients [3]. LeafCheck implements its own four-stage network using that established design idea.")
    doc.add_heading("Contributions", level=1)
    doc.add_paragraph("The project combines an audited image release, a custom CNN trained from random initialization, controlled validation experiments, and a local upload demo. The early plan estimated 133 labels and proposed a pretrained ResNet-18 comparison. The delivered study covers 177 labels and uses Compact177 as the shared-test CNN reference. The interface is a local web app. The team’s earlier Android exploration and requested scope confirmations are recorded in docs/scope_reconciliation.md.")

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
    doc.add_paragraph(f"The paired 256-pixel change versus 224 was {duration['native_size_paired_gain']:+.6f} macro-F1, below the +0.005 promotion threshold. The 256-pixel repeat also failed the duration-trial condition. Two seeds provide limited evidence about run variability.")
    doc.add_paragraph(f"The deployed scoring policy is {policy_name.lower()}. Validation tested logit adjustment [4] and eight-view averaging [5]. Their paired gains were " + ", ".join(f"{policy_names[row['policy']].lower()} {row['mean_gain']:+.6f}" for row in policy["policies"]) + f". Eight-view averaging passed the 500 ms CPU p95 guard at 42.56 ms. The delivered policy reached {validation['macro_f1']:.6f} macro-F1 and {validation['accuracy']:.2%} accuracy on validation.")
    figure(doc, artifact / "learning_curves.png", "Figure 2. The selected training run. The dashed marker identifies the epoch selected on validation.", 5.9)

    heading(doc, "Frozen Test Results")
    table(doc, ["Method", "Correct / total", "Accuracy", "Macro-F1"], [(name, f"{r['correct']:,} / {r['sample_count']:,}", f"{r['accuracy']:.2%}", f"{r['macro_f1']:.6f}")
          for name, r in [("Selected custom CNN", test), ("Compact177 full-data reference", compact), ("Most frequent training class", rule)]], [2.95, 1.35, 1.1, 1.1])
    doc.add_paragraph(f"The original goal was 0.85 held-out macro-F1. The measured result was {test['macro_f1']:.6f}. All three methods use the same 8,425 images. The rule predicts {pretty(rule['rule']['class_name'])}. The model, class order, preprocessing, scoring policy and reference were frozen before evaluation.")
    doc.add_paragraph("The live three-label pilots use different protocols. Create ML’s declared prior training and validation splits overlap the new test by 5,832 and 1,683 images. Its live outputs are exploratory, and the shared-test table omits it.")
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

    if include_field_followup:
        field_followup(doc, docs, freeze, include_crop_head)
    if include_tomato_transfer:
        tomato_transfer(doc, docs, freeze, include_crop_head)

    heading(doc, "Conclusion and Project Management")
    doc.add_paragraph("The custom CNN outscored both primary test references in accuracy and macro-F1. Its lower supplementary accuracy than the conditional frequency rule shows the need for verified, independently collected photos before drawing field-performance conclusions.")
    if include_field_v8:
        pilot = read(docs / "field_optimization/crop_loss_pilot.json")
        unsupported = read(docs / "field_optimization/unsupported_energy.json")["unsupported"]
        accepted = sum(group["accepted"] for group in unsupported.values())
        count = sum(group["count"] for group in unsupported.values())
        doc.add_paragraph(
            "Explicit tomato context correctly classified 15 of 385 screened Hossain photos [17], "
            "compared with zero in automatic mode. The 3.90-point gain missed the five-point target. "
            f"A matched six-epoch crop-loss pilot [18] gained {pilot['field_joint_f1_gain']:.4f} "
            "field macro-F1 and missed its advancement criteria."
        )
        doc.add_paragraph(
            f"The upload rule suggested a label for {accepted} of {count} unsupported Open Images photos [19]. "
            "The demo now lists the covered crops and explains that unfamiliar subjects can receive high scores. "
            "The deployed checkpoint and decision threshold stayed fixed."
        )
    if include_field_v9:
        trial = read(docs / "field_optimization/potato_partial_study.json")
        before = trial["arms"]["control"]["scores"]["field"]
        after = trial["arms"]["field"]["scores"]["field"]
        doc.add_paragraph(
            f"The potato control scored {before['correct']}/{before['sample_count']} on date-held photos "
            f"and the candidate scored {after['correct']}/{after['sample_count']}. "
            f"Adding {sum(trial['screened_source_counts'].values())} field photos [20] "
            f"with exact healthy or combined blight labels gained {trial['field_macro_f1_gain']:.4f} field "
            f"macro-F1 and lost {-trial['master_deltas']['macro_f1']:.4f} on master potato validation. "
            "The candidate missed the registered limits."
        )
    doc.add_heading("Project Management", level=2)
    doc.add_paragraph("Jacob Amey’s assigned role covers data, training, evaluation and model integration. Jacob defined the final scope and authorized the delivery. Demetre’s assigned role covers the frontend and upload interface. The planning decks establish these roles. The contribution record documents available evidence and flags individual implementation and meeting claims for team confirmation. Codex assisted with data preparation, training experiments, evaluation, integration, writing and delivery verification.")
    doc.add_heading("Reproduction and Appendix", level=2)
    doc.add_paragraph("The archive identifies its exact runnable source revision in START_HERE.md. Its demo uses bundled weights. The saved metrics can be recomputed from the archived membership and predictions. Retraining also requires permitted access to the exact original master or frozen images. Follow the restore, verification, cache and pinned training commands in the archived README and data guide.")
    if include_field_followup:
        demo = "field-guidance demo" if include_field_v8 else "potato-context demo"
        doc.add_paragraph("The follow-up study and its manifests are recorded in docs/potato_context_study.md and docs/field_optimization. The " + demo + " was packaged separately from the original course archive.")
    if include_tomato_transfer:
        transfer_location = (
            "Tomato transfer and crop classifier records are in docs/field_optimization."
            if include_crop_head else
            "The matched tomato runs and PlantCity confirmation are recorded in docs/field_optimization/tomato_transfer.md."
        )
        doc.add_paragraph(transfer_location + " The independent follow-up evaluations are complete and excluded from model selection.")
    paragraph = doc.add_paragraph()
    links = [
        ("GitHub project history", "https://github.com/jacob-amey/ML-Final-Project"),
        ("PlantVillage [1]", "https://github.com/spMohanty/PlantVillage-Dataset"),
        ("PlantDoc [2]", "https://github.com/pratikkayal/PlantDoc-Dataset"),
        ("PyTorch source [6]", "https://github.com/pytorch/pytorch"),
    ]
    for name, url in links:
        if paragraph.text:
            paragraph.add_run("  •  ")
        add_link(paragraph, name, url)
    doc.add_paragraph("Dataset SHA-256: " + freeze["dataset_fingerprint"])
    doc.add_paragraph("Selected weights SHA-256: " + freeze["checkpoint_sha256"])
    heading(doc, "References")
    reference_start = len(doc.paragraphs)
    for ref in [
        "[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, ‘Using deep learning for image-based plant disease detection,’ Front. Plant Sci., vol. 7, art. 1419, 2016. doi: 10.3389/fpls.2016.01419.",
        "[2] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, ‘PlantDoc: A dataset for visual plant disease detection,’ Proc. CoDS-COMAD, pp. 249–253, 2020. doi: 10.1145/3371158.3371196.",
        "[3] K. He, X. Zhang, S. Ren, and J. Sun, ‘Deep residual learning for image recognition,’ Proc. CVPR, pp. 770–778, 2016. doi: 10.1109/CVPR.2016.90.",
        "[4] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, ‘Long-tail learning via logit adjustment,’ Proc. ICLR, 2021. https://arxiv.org/abs/2007.07314.",
        "[5] D. Shanmugam, D. Blalock, G. Balakrishnan, and J. Guttag, ‘Better aggregation in test-time augmentation,’ Proc. ICCV, pp. 1214–1223, 2021.",
        "[6] PyTorch, ‘PyTorch source code,’ GitHub. [Online]. Available: https://github.com/pytorch/pytorch",
    ]:
        doc.add_paragraph(ref)
    if include_field_followup:
        for ref in [
            "[7] J. Rashid et al., “Multi-level deep learning model for potato leaf disease recognition,” Electronics, vol. 10, no. 17, 2021, art. 2064. doi: 10.3390/electronics10172064.",
            "[8] N. A. S. Rachin, N. M. Likhon, and M. A. H. Baki, “Tomato Leaf Disease Classification Dataset,” Mendeley Data, v1, 2025. doi: 10.17632/jttrv2w27r.1.",
            "[9] O. Appiah et al., “TOM2024,” Mendeley Data, v1. doi: 10.17632/3d4yg89rtr.1.",
            "[10] T. Wei et al., “A Large-Scale In-the-wild Dataset for Plant Disease Segmentation,” Zenodo, v7, 2025. doi: 10.5281/zenodo.17719108.",
            "[11] M. N. Malik, S. H. Ali, and H. Ali, “Tomato Leaf Disease Classification Dataset in Pakistan,” Mendeley Data, v2, 2026. doi: 10.17632/3mbnb82mxd.2.",
        ]:
            doc.add_paragraph(ref)
    if include_tomato_transfer:
        doc.add_paragraph("[12] M. S. Khan, K. Nisa, I. Ahmad, M. Zubair, and K. Alshammari, “PlantCity: A comprehensive image based on multi crop leaves in Pakistan,” Data Brief, art. 112130, 2025. doi: 10.1016/j.dib.2025.112130.")
    if include_crop_head:
        for ref in [
            "[13] P. M. Kwabena et al., “Dataset for Crop Pest and Disease Detection,” Mendeley Data, v1, 2023. doi: 10.17632/bwh3zbpkpv.1.",
            "[14] N. Tilahun, “Potato Leaf (Healthy and Late Blight),” Mendeley Data, v1, 2020. doi: 10.17632/v4w72bsts5.1.",
            "[15] M. Pellenz, G. B. Zimmermann, A. S. Britto Jr., and Y. M. G. Costa, “Tomato Leaf Image Dataset (TLID/PTLID),” Mendeley Data, v1, 2025. doi: 10.17632/kt64b2kh89.1.",
            "[16] S. Mutuvi et al., “Artemis and Image Safari: Crop imagery datasets spanning globally critical crop species,” agriRxiv, 2026. doi: 10.31220/agriRxiv.2026.00491.",
        ]:
            doc.add_paragraph(ref)
    if include_field_v8:
        for ref in [
            "[17] Hossain et al., “Tomato leaf diseases,” Mendeley Data, v1. doi: 10.17632/93h9p62kg4.1.",
            "[18] L. Bertinetto et al., “Making better mistakes: Leveraging class hierarchies with deep networks,” Proc. CVPR, pp. 12506–12515, 2020.",
            "[19] Google, “Open Images V7,” dataset and image metadata. [Online]. Available: https://storage.googleapis.com/openimages/web/download_v7.html",
        ]:
            doc.add_paragraph(ref)
        if include_field_v9:
            doc.add_paragraph("[20] J. Johnson, “Sample dataset for enhanced field-based detection of potato blight,” Mendeley Data, v1, 2025. doi: 10.17632/pbnw43s6kt.1.")
        for paragraph in doc.paragraphs[reference_start:]:
            paragraph.paragraph_format.space_after = Pt(1.5 if include_field_v9 else 4)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-field-followup", action="store_true")
    parser.add_argument("--include-tomato-transfer", action="store_true")
    parser.add_argument("--include-crop-head", action="store_true")
    parser.add_argument("--include-field-v8", action="store_true")
    parser.add_argument("--include-field-v9", action="store_true")
    arguments = parser.parse_args()
    field_v8 = arguments.include_field_v8 or arguments.include_field_v9
    crop_head = arguments.include_crop_head or field_v8
    tomato = arguments.include_tomato_transfer or crop_head
    build(arguments.artifact, arguments.output,
          arguments.include_field_followup or tomato, tomato, crop_head, field_v8, arguments.include_field_v9)
