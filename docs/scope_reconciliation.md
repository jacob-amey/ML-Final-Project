# CSC-427 scope and course evidence

**Plant Health Classification and an Interactive Application**

Jacob Amey and Demetre

The two team decks identify CSC-427. `Final Project Requirement_Fall2026.docx`
is headed CSC 434. Its proposal, defense, project-management and report
criteria inform these materials. Exact applicability of the weights and
BrightSpace procedure awaits instructor confirmation. The requirements
document describes a two-page team proposal with Introduction, Methods,
Dataset, Goal and Plan. The goal is used to assess completion.
`docs/proposal.md` is the revised working record. The three-class
MobileNetV2 outline at `5616156:docs/proposal.md` is earlier planning evidence.

| Earlier team records | Delivered or changed | Evidence or outstanding confirmation |
| --- | --- | --- |
| 79,086 images and an initial estimate of 133 labels, subject to audit | 84,269 included images and 177 labels after auditing the expanded 90,574-image master | `docs/final_dataset_audit.md` and frozen membership. Faculty acceptance of scope change is pending. |
| Roughly 70/15/15 train/validation/test | Grouped release of 59,008/16,836/8,425, about 70/20/10. Each split retains all 177 labels. The larger validation set supports recipe comparisons on sparse classes. | `data/manifests/leafcheck177_frozen.json`. Many images lack documented leaf or specimen identity. |
| Custom CNN and fine-tuned pretrained ResNet-18 in both decks. MobileNetV2 was contemplated in the earlier outline. | Custom residual CNN trained from random initialization and full-data Compact177 without pretrained weights. Both CNNs and the training-frequency rule use the same held-out images. | Freeze decision `1e65c57`. Instructor acceptance of the comparator change is pending. |
| Target of 0.85 held-out macro-F1 | The selected CNN measured **0.616940** over all 177 labels on the primary test. The original target was missed. | `artifacts/leafcheck177-v1/test_metrics.json`, included in the archive. |
| Try personal and stock photos | Primary evaluation from the audited master and a registered supplementary set of 183 screened PlantDoc photos. The two on-screen examples come from the primary test. A separately sourced Bugwood photo was used only as an upload check. | `external_protocol.json`, `demo_examples.json` and `docs/demo_verification.json`. No verified personal-photo rehearsal or separate stock-photo benchmark. |
| Upload interface with scores and estimated error rate | Local HTML, CSS and JavaScript with a Python HTTP server. Each model returns one inherited condition label and scores. The score ranks labels, and calibration was not fitted. Aggregate errors and scores appear in the evaluation. | `leafcheck-demo/`, `docs/demo_verification.json` and `docs/test_results.json`. |
| React frontend planning and later React Native/Expo Android exploration, including possible treatment suggestions | Working local browser app on macOS. Mobile development and treatment suggestions were exploratory scope. | Deck v3 slide 4 and actual packaged app. Faculty acceptance of the platform change is pending. |
| Measured accuracy, macro-F1, precision, recall, confusion matrices, report and demo | Frozen evaluation, per-class evidence, software checks, illustrated Word report, PowerPoint defense and runnable local archive. | `docs/final_model_report.md`, `docs/submission_quality.json` and `docs/submission_delivery.json`. |

The defense guide uses the peer form’s criteria of significance, methods,
results, validation, clarity and timing. The CSC 434 document lists defense
70% (presentation 25%, methods and results 25%, validation 15%, project
management 5%), collaboration 20% and report 10%. The report follows its
5–10-page, 11-point, normal-margin, 1.15-spacing and bottom-right numbering
format provisionally. The same document lists cross-validation among examples
of testing approaches. The delivered protocol uses fixed grouped splits,
same-test comparisons, a registered supplementary set and software tests.
The build guide’s author and formal authority are unverified.

## Human confirmations

- Instructor or course materials: course-specific rubric, deadline,
  presentation duration, team approval, AI policy, and acceptance of revised
  scope, comparator and platform.
- Jacob or Demetre: formal proposal receipt, individual task records and
  meeting history, personal rehearsal, and actual submission or receipt.
- Team and data owners: permitted access to the exact master images for a
  recipient who needs to restore and retrain. The local package supplies the
  manifest and runnable materials for the demo and score audit.

If the instructor requires a transfer-learning comparison or mobile app,
confirm that requirement and scope it separately. Any new model evaluation
needs a separately defined independent test because the original holdout
has been opened.

## Records reviewed

- `../AI&ML/Final Project Requirement_Fall2026.docx`
- `../AI&ML/project evaluation form_Fall2026_student.docx`
- `../AI&ML/Final Project Presentation CSC-427-2.pptx` and `-3.pptx`
- `AI Project Build Guide.docx`
- `5616156:docs/proposal.md`, current `docs/proposal.md` and `docs/contributions.md`
