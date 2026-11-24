# LeafCheck field expansion

This is a follow-up to the 177-label CSC-427 submission and its
[upload reliability study](field_reliability_study.md).
The original custom CNN was initialized randomly. The experiments here start
from that in-house checkpoint and update its trainable weights. Original
preprocessing, the 177-label order, and the frozen course artifacts remain the
comparison reference. No externally pretrained model is used.

The *AI Project Build Guide* sections 3–5 and 7–9 motivate traceable examples,
related-image groups, a frequency baseline, validation selection, a final test,
and a reproducible demo. The [course scope record](scope_reconciliation.md)
explains the provisional applicability of the available CSC 434 rubric.

## Sources and split protocol

| Source | Role | Label policy |
| --- | --- | --- |
| [Bapari, Zitu and Ummehani, Bangladesh tomato leaves](https://data.mendeley.com/datasets/74zvtzx9pr/1) | Whole collection confined to training. The publisher describes field photography and leaf crops, and some reviewed photos have plain backgrounds. | Use bacterial spot, healthy and Septoria. Generic blight, leaf curl and mosaic labels do not identify the corresponding conditions precisely enough. |
| [Huang and Chang, Taiwan tomato leaves](https://data.mendeley.com/datasets/ngdgg79rzb/1) | Whole Taiwanese collection confined to the new tomato test. Use only its preprocessed originals, with mixed single and complex backgrounds. | Use bacterial spot, healthy and late blight. Keep all 177 outputs eligible. |
| [Laizer, Mduma and colleagues, Tanzanian potato leaves](https://zenodo.org/records/8286529) | Deterministic sample from the three pinned publisher ZIP archives. Where enough dates exist to cover every label, reserve complete EXIF days for development validation. | Use potato early blight, healthy and late blight. Unknown days can contribute only to training. |
| [Sharma and Raheman, field potato corpus](https://zenodo.org/records/22059910) | Download and provenance inspection only. | The archived photographs have no per-image disease labels. Exclude from condition fitting and scoring. |
| [Bousset-Vaslin, Inrae potato fragments](https://doi.org/10.57745/D6DVD5) | Metadata screened only. The archive's public file server repeatedly reset the transfer. | The table documents candidate early/late labels, but the image bytes were unavailable. |
| [Zhuo et al., M-TLD tomato collection](https://github.com/zhuojiaxiong6/DETR#-dataset) | Source availability checked. The authors' repository directs users to request access. | The archive and per-image annotations were unavailable for local screening. More than one annotation per image is common in the published study. |

The three admitted sources use CC BY 4.0. [Exact source and archive records](../data/manifests/leafcheck_field_v2_sources.json)
include attribution and checksums. Labels remain inherited from the publishers.
The builder screens exact bytes, decoded RGB fingerprints invariant to rotations
and reflections, and perceptual hash distances against 90,574 original master
records, 183 registered PlantDoc images, all included blight v1 images, and 218
registered PLDD-UP images. It then collapses repeat views, excludes conflicting
labels, verifies class indices, and hashes each retained original. These checks
can miss cropped copies and related photos without known specimen IDs. Whole
source and capture-day boundaries do not prove independent specimens.
Source image bytes and all existing frozen evaluation results remain separate.

The tomato pilot kept 419 training images (66 bacterial spot, 265 healthy,
88 Septoria) and 314 blind test images (110 bacterial spot, 106 healthy,
98 late blight). Thirty-three candidates failed the audit. Source-level
evaluation can reveal transfer across collections and is limited to the
labels and conditions represented in those collections.

The expanded release kept 1,829 new training images and 211 date-held
potato validation images. The 1,410 Tanzanian training images span 41
recorded days. Validation spans 13 disjoint days. Each retained Tanzanian
file had EXIF date information. The largest single training day contributes
336 photos. This is an observation about capture-day concentration.
Evidence for independent farms or specimens is limited. The earlier v1
Tanzanian additions came from reported capture dates in December 2024.
The new study's 2023 source dates are disjoint from those additions.

The [sampled source registry](../data/manifests/leafcheck_field_v2_sampled_sources.json)
pins the early-blight archive and exact 128 MiB prefixes for the healthy
and late-blight archives. All local ZIP members used from the prefixes
passed compressed size, decoded size and CRC checks. A partial final
member cannot enter the release. The SHA-256 pins refer to the received
prefix bytes and the publisher MD5s refer to the whole archives. The
registered sample uses 600 candidates per potato class before overlap
screening. Healthy and late-blight candidate pools are constrained to those
archive prefixes, so they are not a random sample of the full collections.

## Matched treatment

The [pilot recipe](../configs/leafcheck_field_v2.json) adds the new field
photos to an identical master replay and previous supplement. The control
takes the same number of historical training photos in the same label order
in their place. Both initialize from the same in-house weights and use
matched augmentations, steps, seeds, trainable layers, and evaluation. The
teacher stays fixed and regularizes the old-domain minibatch items using
its own soft scores. Batch-normalization statistics stay fixed. The control
distinguishes the value of new photos from the benefit of revisiting older
supplement images.

Master validation macro-F1 across 177 labels is primary. The field slice
constraint requires a gain of 0.06 versus the incumbent and a gain of 0.02
versus the matched control. Maximum master losses are 0.002 macro-F1 and
0.005 accuracy. Diagnostics include wrong-crop and early-to-late and
late-to-early counts. The best eligible single-view epoch receives the
deployed eight-view check. If every epoch fails, the strongest master
single-view epoch is scored for diagnosis. The previous 8,425-image test,
PlantDoc score, and PLDD-UP score do not steer these decisions.

The actual Sheikh date-held split contains 56 early-blight and 177 healthy
tomato images. Its macro-F1 uses these two represented classes, with all 177
predictions eligible. The Banu slice has 39 images across potato late blight,
tomato healthy and tomato late blight. The pilot recipe's reference to three
Sheikh labels describes the broader supplement rather than actual validation
support. Banu is recorded separately and carries no selection weight.

| Pilot, eight-view scoring | Master validation correct | Master macro-F1 | Sheikh macro-F1 | Sheikh wrong crop |
| --- | ---: | ---: | ---: | ---: |
| Incumbent | 15,111/16,836 | 0.649221 | 0.000000 | 229/233 |
| Matched control, epoch 4 | 14,924/16,836 | 0.642097 | 0.573392 | 79/233 |
| Added tomato photos, epoch 3 | 14,797/16,836 | 0.628517 | 0.624175 | 59/233 |

The added data gained 0.050783 field macro-F1 over the matched control and
missed the broad validation limits. Its 28/56 early-blight-to-late-blight
errors show that resolving crop identity leaves substantial condition
confusion on this development slice. It is an exploratory checkpoint.

A [follow-up recipe](../configs/leafcheck_field_v2_full.json) replays all
59,008 master training examples over four epochs in each arm. A separately
[registered interpolation screen](../configs/leafcheck_field_v2_interpolation.json)
tests whether smaller weight changes can retain broad validation performance
with the same deployed network size and compute. This borrows the weight
interpolation idea from Wortsman et al. and applies it to the two states of
our own CNN. [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)

The tomato interpolation screen retained 0.648734 master macro-F1 at a
6.25% dose of the field-trained weights. At that dose the Sheikh field
macro-F1 stayed at 0. A larger dose improved the single-view field score
while falling below the master limits. No checkpoint from that screen
earned promotion.

## Expanded replay results

Both arms completed four epochs. Each procedure replayed all 59,008 master
training images once across those epochs. The epoch-two field checkpoint
was selected for the diagnostic comparison by the primary validation
metric. That checkpoint had replayed 29,504 master images after starting
from the already full-data incumbent.

| Eight-view scoring | Master correct | Master macro-F1 | Sheikh tomato correct | Mduma potato correct | Mean field macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Incumbent | 15,111/16,836 | 0.649221 | 0/233 | 7/211 | 0.025132 |
| Matched control | 14,972/16,836 | 0.643086 | 65/233 | 16/211 | 0.269331 |
| Expanded field data | 14,917/16,836 | 0.633074 | 134/233 | 143/211 | 0.674532 |

Mduma wrong-crop predictions fell from 195/211 to 32/211. Among potato
late-blight references, 32/100 were predicted as potato early blight,
compared with 9/100 for the incumbent. The reverse direction changed
from 0/61 to 2/61. More predictions named the correct crop while early
and late symptoms remained hard to separate. The broad macro-F1 fell
by 0.016148 and accuracy fell by 1.15 percentage points. The direct
field-adapted candidate failed both deployment limits.
[Run and environment](field_expansion/replay_run.json) ·
[Checkpoint and per-source results](field_expansion/replay_decision.json)

## Guarded field model and frozen test

Weight interpolation in the expanded study increased broad validation
macro-F1 to 0.652578 at a 25% field weight contribution. Its field mean
macro-F1 reached 0.062134, below the predeclared gain from 0.025132.
The guarded inference experiment used the field-trained prediction only
when the original model score was below its existing withholding limit
and the field model score was at least 0.90. On master validation it
changed 13 of 16,836 predictions, with four fewer correct classifications
and ten more wrong suggestions. It met the configured limit of 0.001
additional wrong suggestions per master upload. All scores remain
uncalibrated.

| Validation scoring | Master accuracy | Master macro-F1 | Sheikh correct | Mduma correct | Mean field macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original model | 89.754% | 0.649221 | 0/233 | 7/211 | 0.025132 |
| Guarded field experiment | 89.730% | 0.648008 | 32/233 | 48/211 | 0.259522 |

The guarded policy passed the validation limits and was frozen before
opening the separate 314-photo Taiwan test. The original and guarded
models both classified **38/314** correctly (12.10%, macro-F1 0.114630).
The training-frequency comparator always picked bacterial spot among the
three represented labels and reached 110/314 (35.03%, macro-F1 0.172956).
The guarded model changed three predictions and no correctness outcomes.
Both CNN policies correctly identified 38/98 late-blight photos and no
healthy or bacterial-spot photos. The Taiwan release contains no mapped
early-blight photos for the reverse confusion denominator. Both policies
predicted tomato early blight for 2/98 late-blight photos. The largest
uncertainty comes from applying this result to other collections.
[Frozen test protocol](field_expansion/taiwan_freeze.json) ·
[Frozen test results](field_expansion/taiwan_metrics.json)

The original verified model remains the normal LeafCheck startup.
The guarded pathway is opt-in for research and classroom demonstration.
The cross-source results do not establish a general improvement on
arbitrary uploads. Potato validation remained within the publisher's
collection. It tests unseen dates and still needs confirmation on an
independent labeled potato collection. Keep the Taiwan result frozen
when making any subsequent experiments.

## App verification

Ten original diagnostic or validation uploads produced identical ordered
scores through the API and standalone CPU inference. Thirty-seven Banu
validation images exceeded a client or server upload limit. The demo
now shows both limits before selection and explains how to handle an
oversized file. Its managed archive retains the previously frozen model.
[Archive receipt](field_expansion/demo_archive.json) ·
[Fresh extraction verification](field_expansion/archive_verification.json) ·
[Browser and API verification](field_expansion/upload_verification.json) ·
[Guarded upload checks](field_expansion/experimental_upload_verification.json)

## Reproduce

From the project root with the pinned research environment, original field
supplement, master release and its 224-pixel training/validation cache:

```sh
.venv/bin/python scripts/prepare_field_expansion.py fetch --tomato-only
.venv/bin/python scripts/prepare_field_expansion.py build --tomato-only --output data/processed/leafcheck_field_v2_tomato
.venv/bin/python scripts/prepare_field_expansion.py verify --output data/processed/leafcheck_field_v2_tomato
.venv/bin/python -m leafcheck.field_expansion --data data/processed/leafcheck_field_v2_tomato --run runs/field_v2/tomato_reproduction
```

To reproduce the expanded run, use the sampled registry and expanded recipe.

```sh
.venv/bin/python scripts/prepare_field_expansion.py fetch --registry data/manifests/leafcheck_field_v2_sampled_sources.json
.venv/bin/python scripts/prepare_field_expansion.py build --registry data/manifests/leafcheck_field_v2_sampled_sources.json --output data/processed/leafcheck_field_v2_sampled
.venv/bin/python -m leafcheck.field_expansion --data data/processed/leafcheck_field_v2_sampled --config configs/leafcheck_field_v2_full.json --run runs/field_v2/expanded_reproduction
```

For the archived study, the exact result directories are
`runs/field_v2/replay_full`, `runs/field_v2/replay_interpolation`,
`runs/field_v2/replay_routing`, and `runs/field_v2/frozen_taiwan`.
The route tool takes `--release`, `--parent`, `--output` and `--plan`
from the corresponding release, matched run, new output directory and
`configs/leafcheck_field_v2_routing.json`. The separate
`leafcheck.evaluate_field_expansion` module freezes the parent and the
selected routing receipt through `--decision` and `--routing`.
Evaluation reads that freeze and rejects changed weights, preprocessing
or test membership. The tracked receipts allow inspection without
reopening the test. A fresh dataset build reproduced the saved manifest
and release record byte for byte.

To resume, supply the original configuration, dataset and run directory
and add `--resume`. The trainer saves model, optimizer, RNG state and history
at the end of each epoch. An interrupted partial epoch is replayed.
The paired and interpolated runs use different output directories.
Model, cache, and image files are local. The Git sources and release records
identify them and explain restoration.

The tested study package is `runs/field_v2/frozen_taiwan/`.
From the project root, use an unused port to launch its optional upload
mode. Normal startup omits the environment setting.

```sh
LEAFCHECK_FIELD_STUDY="$PWD/runs/field_v2/frozen_taiwan" \
  ./leafcheck-demo/Launch\ LeafCheck.command --port 8770
```

The app checks the frozen reference, new weights, class order,
preprocessing and evaluation receipt before enabling the study.
The existing archived test table describes the original model.
The experiment does not require training images for inference.
