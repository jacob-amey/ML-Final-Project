# LeafCheck final model requirements and execution record

## Verified sources and scope

The current project is CSC-427, **Plant Health Classification and an Interactive
Application**, as identified by the user. The earlier course rubric pasted in
the project conversation (labeled CSC 434 there) requires implementation,
comparative validation/testing, a working demonstration and Q&A, a documented
GitHub repository, a two-page proposal, a 5–10 page final report, and explicit
individual contributions. The local `AI Project Build Guide.docx` additionally
requires a most-common-training-label rule, validation-led revision, frozen final
testing, and a fresh demo with one success, one mistake, scores and future work.
The guide recommends approximate 70/15/15 splits; the audited grouped 70/20/10
release is retained because it already provides separate development and testing.

The user's current instruction establishes a custom CNN, randomly initialized,
with all 177 labels. Standard PyTorch layers and automatic differentiation are
allowed by the user's stated definition. The old three-class transfer-learning
proposal is superseded. No additional local course AI-assistance restriction was
found; institutional policy and approval of the two-person team are not verified.
The earlier pasted notebook walkthrough is available. The original complete
rubric is available in the conversation rather than a separate local attachment.

## Requirement and evidence checklist

| Requirement | Implementation/evidence | State at start |
| --- | --- | --- |
| 177-class custom CNN, no pretrained weights | `leafcheck/model.py`, configs, initialization and run records | Pending |
| Audited, reproducible train/validation/test data | Release manifest, mapping, `docs/final_dataset_audit.md` | Verified 84,269 images; full release verification rerun |
| Traceable IDs, whole related groups, prior exposure precautions | Release evidence, frozen dataset fingerprint | Existing checks verified; residual missing-ID risk documented |
| Training-only learned statistics and augmentation | Shared preprocessing, cache and training configuration | Pending |
| Finite gradients, tiny-set learning, checkpoint/reload/resume | Diagnostic entry point and tests | Pending |
| Full designated training split | Full-data run logs with 59,008 examples/epoch | Pending |
| Controlled experiments and simple rule | Validation records and experiment register | Pending |
| Selection independent of test | Validation macro-F1 rule, freeze record, separate evaluator | Rule declared below |
| Final accuracy/counts, macro-F1, class metrics and figures | Frozen evaluation artifacts and summary | Pending |
| Honest comparison with earlier models | Exposure/coverage audit and website text | Pending; earlier models are not fair held-out baselines |
| Standalone packaged real model in website | Explicit artifact reference, parity and fresh-process integration tests | Pending |
| Invalid upload handling and complete browser rehearsal | API tests and observed browser run | Pending |
| Reproducible versions, seeds, device, code and artifact hashes | Per-run environment, configs, manifests and checkpoints | Pending |
| Git milestones and documented assistance/contributions | Work log, commits and contribution record | In progress |
| Report, literature, proposal and presentation | Existing proposal needs scope correction; final methods/results documentation | Formal submission and student presentation remain student responsibilities |

## Declared model selection and stopping policy

Primary checkpoint and configuration metric: **validation macro-F1 across all
177 classes**, including zero F1 when a class is never correctly predicted.
Break ties within 1e-6 by lower unweighted validation cross-entropy, then earlier
epoch. Across effectively tied configurations prefer the smaller/faster model.
Test predictions, metrics, images of mistakes and thresholds must not enter this
selection process. No confidence threshold is planned; display scores with an
explicit uncertainty explanation.

Benchmark the actual hardware and pipeline first. Establish a full-data compact
CNN reference, then test justified capacity and imbalance changes one at a time.
Each run records its planned epoch budget, minimum training, early-stop patience
and learning-rate policy before execution. Keep the best checkpoint safe. Give
the leading configuration further training or an independent seed when timing
allows. Stop after validation gains flatten despite learning-rate reduction, or
when the measured cost of further runs is unreasonable for this local final
project; document the actual reason. A gain of 0.005 macro-F1 is a practical
signal for a meaningful configuration change, not a claim of statistical
significance. Rare-class supports and correlated views limit all comparisons.

## Initial repository and hardware state

Parent repository: `dev` at `30eb970`, with staged dataset preparation changes in
four files from the previous work. These are relevant and will be committed as
their own reviewed milestone. The nested `leafcheck-demo` is an independent Git
repository at `1d17fb8` with existing local migration/bundling/UI changes. Those
edits are preserved and identified separately from new final-model integration.
Initial status, staged/unstaged patches, hashes and copies of touched demo files
are saved under `runs/final177/preflight/`. Work proceeds on
`codex/final-leafcheck177` in both repositories. Unrelated changes will not be
reset or included in model commits.

Hardware verified: Apple M2 Max, 30 GPU cores, 32 GB unified memory, working MPS;
about 332 GB free storage. Existing environment: Python 3.14.7, PyTorch 2.15.0,
torchvision 0.29.2, Pillow 13.0.0. Seeds support reproducibility; device/backend
floating-point differences can prevent bitwise identity across hardware.

Official API references checked during implementation:
[MPS RNG/state APIs](https://docs.pytorch.org/docs/main/mps.html),
[cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html),
[AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html).

## Resolution protocol amendment before held-out evaluation

On November 16 the user confirmed no hard deadline and prioritized a defensible
result. `final_selection_plan.json` records one additional 160-pixel check, with
architecture, data, seed, weighting and training settings fixed. The observed
128-pixel seed range is 0.005365 macro-F1. A seed-434 gain at least that large
triggers a 160-pixel seed-435 repeat. Accept 160 only if its two-seed mean gain
over 128 is at least 0.005; otherwise prefer 128's lower compute cost. This
acceptance rule was declared before any 160-pixel validation results. Checkpoint
F1/loss selection stays unchanged. This is a practical effect-size rule, not a
claim of statistical significance.

## Resource and resolution policy amendment on resumed work

The user's later instruction explicitly removed time and resource intensity as
limiting preferences. The earlier cost-driven 160-pixel acceptance rule above is
historical and superseded by the timestamped amendments in
`final_selection_plan.json`. Competitive resolutions receive a matched seed
repeat, and paired-seed validation macro-F1 informs the choice; small differences
remain tentative. Mild brightness/contrast augmentation is the next declared
single-setting trial. The stopping decision must use validation plateaus,
reproducibility and data limitations; no test prediction may influence it.

## Verified development progress, November 17

The historical “State at start” column above is not a final delivery claim.
Current completed development evidence includes:

| Requirement | Verified evidence now | Remaining work |
| --- | --- | --- |
| Custom random-initialization177 outputs | `leafcheck/model.py`; completed full-data configs/status/checkpoints | Choose/freeze the accepted final recipe |
| Shared deterministic input preparation | Saved train-only statistics; `preprocessing.py`; exact row metadata checks at128/160/224/256 | Verify selected bundle/API parity |
| Pipeline can learn and save/reload | `runs/final177/diagnostics/checks.json`:12/12 tiny training examples after40 updates; finite gradients and parameter changes; reload passed | Final selected-artifact test suites |
| Full training split and validation objective | Every full epoch visits59008; run histories/status, `docs/experiments.csv` | Finish active scaling procedures |
| Controlled comparisons and stability | Paired128/160, illumination and decay evidence; completed wider screen; first224 procedure complete/reloaded | Finish224 repeat and conditional native-size check |
| Dataset freeze and test seal | Fingerprint4287764ba029d96abaa0532d7e1d14d2656cadc774f445be7785b03b69d2903c; `test_seal_record.json` | Freeze selected checkpoint, then evaluate once |
| Honest comparison protocols | `comparison_protocol.json`: full-data Compact177 fair reference; old pilots separate; Create ML exposure limitation | Record frozen shared-test scores |
| Reproducible integration contract | `model_handoff.md`, explicit bundling/inference entry points, minimal app environment | Install evaluated artifact, real API/browser/archive rehearsal |

The positive first224 result and training-only native geometry justified the
prospectively declared conditional256 check. Latest rules are timestamped in
`final_selection_plan.json`; they supersede historical resource ceilings.
No final test predictions or supplementary predictions have informed selection.

## Final delivery verification

| Requirement | Completed evidence |
| --- | --- |
| Custom random-initialization CNN and full designated train split | `configs/residual177_224_seed434.json`, `docs/selected_architecture.csv`, selected run history and checkpoint. The procedure completed 63 full epochs of 59,008 images and chose epoch 53. |
| Train-only statistics, augmentation and reproducible resume | Frozen preprocessing record, pipeline diagnostics, CPU and MPS interrupted-resume tests, and the logged 256-pixel recovery. |
| Grouped reproducible dataset with all 177 labels | Frozen release fingerprint and audit, manifest in the local submission archive, exact restoration procedure in `data/README.md`. |
| Validation-controlled selection and independent final testing | `docs/final_model_selection.json`, `docs/test_seal_record.json`, freeze manifest, paired comparisons and `docs/final_model_report.md`. |
| Honest comparison and detailed measured results | Frozen test metrics, predictions, per-class support and confusion figures, same-split Compact177 and frequency rule, and separately scoped PlantDoc results. |
| Final-model deployment and live upload | `docs/test_results.json` records 9 pipeline and 15 website tests. `docs/demo_verification.json` records actual uploads, outcomes and error handling. |
| Fresh extraction and documented launch | `docs/archive_verification.json` records the launched four-model process using the packaged app-only environment, real image predictions and invalid-input responses. |
| Report, presentation and reproducibility | Eight-page report, defense slides, executed review notebook, local source snapshot and artifact records in the combined submission. |
| Team approval and formal course submission | The assigned model and app roles and actual assisted work are recorded in `docs/contributions.md`. Team names, approval, additional institutional AI rules and submission status require team confirmation. |

## Field update report checklist

The local project requirement identifies CSC 434 and the team decks identify
CSC-427. The available report requirements have been applied provisionally.
The original combined submission remains the archived course snapshot.

| Requirement | Evidence in the editable Field Study report |
| --- | --- |
| Introduction, related work and original contributions | Page 1 gives the task, prior literature and delivered contribution. |
| Methods, source integrity, training and selection | Pages 2–4 give the frozen split design, architecture and train-only preprocessing, and explain seed and recipe comparisons. |
| Results, figures and comparative validation/testing | Pages 4–8 show learning curves, directed confusions, paired validation, three methods on the same reserved test, supplementary studies and the matched tomato comparison with its independent PlantCity check. |
| Discussion, conclusion and future improvement | Pages 6–9 explain sparse support, dataset shift, uncalibrated scores and the need for independent field data. |
| Project management and individual contributions | Page 9 distinguishes assigned roles, evidenced decisions and assistance. Student implementation details and meeting activities still require confirmation. |
| IEEE references, repository link, dataset and source links | Pages 9–10 give the GitHub link, source and restoration pointers, dataset version, weights hash and numbered references. Independent access to the full merged master remains unverified. |
| Length and formatting | The [verification record](field_optimization/report_tomato_verification.json) covers ten rendered pages, 11-point body, 1.15 line spacing, one-inch margins and bottom-right page numbering. |
| Single team BrightSpace submission | The team owns submission and confirmation. The separate field update is a local editable report. |
