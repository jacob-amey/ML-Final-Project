# LeafCheck final model requirements and execution record

## Verified sources and scope

The CSC 434 rubric pasted in the project conversation requires implementation,
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
