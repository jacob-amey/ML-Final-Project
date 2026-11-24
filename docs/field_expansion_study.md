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

## Reproduce

From the project root with the pinned research environment, original field
supplement, master release and its 224-pixel training/validation cache:

```sh
.venv/bin/python scripts/prepare_field_expansion.py fetch --tomato-only
.venv/bin/python scripts/prepare_field_expansion.py build --tomato-only --output data/processed/leafcheck_field_v2_tomato
.venv/bin/python scripts/prepare_field_expansion.py verify --output data/processed/leafcheck_field_v2_tomato
.venv/bin/python -m leafcheck.field_expansion --data data/processed/leafcheck_field_v2_tomato --run runs/field_v2/tomato_reproduction
```

To resume, supply the original configuration, dataset and run directory
and add `--resume`. The trainer saves model, optimizer, RNG state and history
at the end of each epoch. An interrupted partial epoch is replayed.
The paired and interpolated runs use different output directories.
Model, cache, and image files are local. The Git sources and release records
identify them and explain restoration.
