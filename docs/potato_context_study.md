# LeafCheck study of crop context and new field images

The CSC-427 delivery uses a custom, randomly initialized 177-label CNN. This
follow-up keeps its original checkpoint, class order, preprocessing, and
reported test results. The *AI Project Build Guide*, sections 3–5 and 7–9,
motivates the grouped comparisons, validation selection, reserved checks and
working demo. The [course scope record](scope_reconciliation.md) explains the
provisional applicability of the available CSC 434 rubric.

## What went wrong on new photos

The original photo files and independently verified diagnoses from Jacob's
report were unavailable for replay. The [upload audit](upload_blight_study.md)
used labeled substitutes and found that the website and standalone CNN agreed
on accepted photos. On three-label Okara potato images the 177-label CNN named
a different crop in 3,288 of 3,407 predictions. Misreading the crop accounted
for most observed automatic errors on this collection. The model also confused
conditions after the crop was supplied. These results show a source transfer
problem. Class counts alone do not establish its cause.

## Validation decisions

The full automatic ensemble averages the two original 224-pixel CNNs' D4
probabilities. On the same 16,836 master validation images it scored
15,158 correct (90.0333%, macro-F1 0.657068), versus 15,111 (89.7541%,
macro-F1 0.649221) for the original. The 0.007846 gain passed the declared
0.005 validation criterion. Wrong-crop errors fell from 956 to 917.
On grouped field validation, both missed every one of the 233 Sheikh tomato
images. The ensemble got 12/211 correct on Mduma potato, versus 7/211.
Keep this candidate as a validation-only alternative for the full 177-label
task. The Okara check supports just three potato labels and the original
177-label test had already been opened.

Two custom potato CNNs per recipe start with random weights at 128 pixels.
Each field run used 1,771 unique master training images and 1,429 screened
field training images. Its matching control replaced the field images with
same-label resampled master images. The 211 Mduma development photos use
held-out dates. The master potato validation slice has 502 images. The
field specialists' mean Mduma macro-F1 was 0.870840 versus 0.409394 for the
control, but neither seed met the master noninferiority limit. The preregistered
50:50 probability blend passed the master limits for both seeds. Its mean
Mduma macro-F1 was 0.686127, against 0.283511 for the matched blend control
and 0.245694 for the original CNN given potato. The frozen blend used the
stronger field-validation seed, 435. These figures use development data.

[Validation decision and per-source errors](potato_context/validation.json) ·
[Scratch CNN decisions and checkpoint hashes](potato_context/potato_training.json)

## Frozen Okara check

The public mirror contains 4,072 candidate images. The original paper reports
4,062. A later paper identifies the same mirror and its 4,072 images. The
[pinned source record](../data/manifests/leafcheck_okara_v1_source.json)
identifies the exact downloaded archive and license listing. The preparation
screened bytes, decoded pixels and perceptual hashes against 90,574 original
master records and the existing PlantDoc, PLDD-UP, blight v1 and field v2
releases. It excluded 187 possible previous visual matches, 70 repeat views
and 408 members of potential label-conflict groups. Similarity screening can
miss related images or group visually similar leaves. The retained 3,407
images include 1,265 early blight, 854 healthy and 1,288 late blight. The
publisher's labels are inherited. Specimen IDs are unavailable. A contact
sheet showed mostly isolated or cropped leaves on simple backgrounds, so
the check measures transfer to another collection of potato leaves with
limited evidence about garden uploads.
[Source paper](https://doi.org/10.3390/electronics10172064) ·
[Mirror](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld) ·
[Later study citing the mirror](https://pmc.ncbi.nlm.nih.gov/articles/PMC9272756/)

All methods and their weights were fixed before this check. Macro-F1 here
averages the three represented labels. The automatic models can still
predict any of 177 labels. The known-potato modes receive the crop as an input.

| Method and scope | Correct / 3,407 | Accuracy | Macro-F1 | Wrong crop |
| --- | ---: | ---: | ---: | ---: |
| Original automatic CNN, 177 labels | 74 | 2.17% | 0.038106 | 3,288 |
| Two-seed automatic ensemble, 177 labels | 110 | 3.23% | 0.054729 | 3,253 |
| Original CNN given potato, three labels | 1,971 | 57.85% | 0.543382 | 0 |
| Field-trained scratch CNN given potato | 1,401 | 41.12% | 0.300115 | 0 |
| Original and specialist blend given potato | 1,535 | 45.05% | 0.362274 | 0 |
| Most frequent potato training condition, late blight | 1,288 | 37.80% | 0.182890 | 0 |

| Method | Early → late / all early | Late → early / all late |
| --- | ---: | ---: |
| Original automatic CNN | 2 / 1,265 | 35 / 1,288 |
| Two-seed automatic ensemble | 0 / 1,265 | 42 / 1,288 |
| Original given potato | 170 / 1,265 | 599 / 1,288 |
| Scratch field specialist | 87 / 1,265 | 1,119 / 1,288 |
| Known-potato blend | 62 / 1,265 | 1,063 / 1,288 |

Under crop restriction, every error is within the crop by construction.
The scratch model and the blend improved on the Tanzanian validation dates
but lost to the original crop-conditioned predictor on Okara. The blend
reduced early-to-late errors while making many more late-to-early errors.
Keep both field-trained policies experimental. There is no tomato directional
estimate in this potato-only set. See the [prior blight analysis](upload_blight_study.md)
for tomato denominators.

[Freeze](potato_context/okara_freeze.json) ·
[Reconciled metrics, per-class support and examples](potato_context/okara_metrics.json)

## TOM2024 field-training comparison

This separate experiment tested whether additional West African field images
could improve the full 177-label model beyond matched rehearsal. The release
contains 578 screened publisher files, including 255 healthy tomato, 73 tomato
late blight and 250 healthy maize. All enter training. Eight candidates were
excluded as possible repeated views. The source uses CC BY 4.0. The source
and per-file hashes are pinned, and a fresh build reproduced the manifest
and release receipt byte for byte. [TOM2024 dataset](https://data.mendeley.com/datasets/3d4yg89rtr/1) ·
[Data paper](https://doi.org/10.1016/j.dib.2025.111357) ·
[Release seal](../data/manifests/leafcheck_tom2024_v1_release.json)

Both arms continued from the in-house checkpoint, with four epochs that
collectively used the entire 59,008-image master training split once. Both
arms used the same earlier field training releases, mild augmentation and
teacher loss from the original CNN. The control filled the new image
positions with same-label master training examples. Development used the
same 16,836 master validation images and grouped Sheikh and Mduma slices.
The field column is the equal mean of the two slice scores.

| D4 policy | Correct / 16,836 | Master accuracy | Master macro-F1 | Field mean macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Original checkpoint | 15,111 | 89.75% | 0.649221 | 0.025132 |
| Matched control, epoch 2 | 14,989 | 89.03% | 0.637194 | 0.296866 |
| TOM2024 treatment, epoch 1 | 14,977 | 88.96% | 0.635150 | 0.141294 |

Neither arm passed the master eligibility screen. The selected treatment
trailed the matched control on the master and on both field slices. Later
treatment epochs reached higher field scores with a master regression that
remained beyond the declared tolerance. Tomato late blight recall on the
master slice was 369/414 in both arms versus 370/414 for the original.
These results reject this replay recipe. They leave open the benefit of a
different balanced collection or training protocol. No reserved set was
run for the failed candidate.
[Run configuration, checkpoint hashes and complete validation trajectory](potato_context/tom2024_matched.json)

Reproduce with the [pinned recipe](../configs/leafcheck_tom2024_training_v1.json)
after restoring the [data releases](../data/README.md).

```sh
.venv/bin/python -m leafcheck.field_expansion \
  --data data/processed/leafcheck_tom2024_v1 \
  --config configs/leafcheck_tom2024_training_v1.json \
  --run runs/field_v3/tom2024_matched
```

Add `--resume` with the same arguments and the exact trainer revision
`f79bb4b` to continue a compatible checkpoint. For an independent
replication, use a new run directory.

## Reserved Bangladesh tomato check

This separate check tested whether the validation-selected two-seed gain
transferred to another source that contains both tomato blights. A fixed
publisher-ID hash selected up to 125 images per compatible label, including
all 112 late blight candidates. The 604 verified publisher originals yielded
508 evaluation images after removing 23 possible cross-release matches, 27
candidate repeat views, and 46 members of potential label-conflict groups.
Specimen IDs and independent diagnoses are unavailable. The authors report
collection across agricultural regions. The reviewed contact sheet showed
isolated leaves on light backgrounds, so this is a source-transfer check
with limited relevance to whole-plant garden uploads.
[Publisher and CC BY 4.0 listing](https://data.mendeley.com/datasets/jttrv2w27r/1) ·
[Release seal](../data/manifests/leafcheck_bangladesh_tomato_v1_release.json)

The evaluation plan and exact membership were frozen before inference.
Every method uses the same 508 image IDs. Macro-F1 averages the five source
labels, including 105 early blight and 95 late blight examples. Automatic
mode can return any of 177 labels. For the known-crop diagnostic, tomato is
supplied and all ten existing tomato labels stay eligible.

| Method | Correct / 508 | Accuracy | Macro-F1 | Wrong crop |
| --- | ---: | ---: | ---: | ---: |
| Original automatic CNN | 1 | 0.20% | 0.004040 | 486 |
| Two-seed automatic ensemble | 0 | 0.00% | 0.000000 | 491 |
| Original CNN given tomato | 56 | 11.02% | 0.098884 | 0 |
| Two-seed ensemble given tomato | 59 | 11.61% | 0.127929 | 0 |
| Automatic training frequency rule, grape black rot | 0 | 0.00% | 0.000000 | 508 |
| Known-tomato training frequency rule, yellow leaf curl virus | 107 | 21.06% | 0.069593 | 0 |

| Method | Early → late / all early | Late → early / all late |
| --- | ---: | ---: |
| Original automatic CNN | 1 / 105 | 3 / 95 |
| Two-seed automatic ensemble | 1 / 105 | 2 / 95 |
| Original given tomato | 26 / 105 | 15 / 95 |
| Two-seed ensemble given tomato | 20 / 105 | 16 / 95 |

The original automatic model predicted another crop on 98/105 early blight
and 90/95 late blight photos. Given tomato it identified 10/105 and 39/95,
respectively. The known-crop classifiers spread correct predictions across
multiple labels, which improves macro-F1 over the rule while yielding lower
top-1 accuracy. The ensemble's higher master validation score did not
translate to accurate automatic answers here. On this source an automatic
late-blight mistake can remain a condition mistake even when the crop is
supplied. The two documented late-blight examples preserve the inherited
labels and each model's prediction. The deployed app offers potato context.
The tomato scores are research diagnostics.
[Freeze](potato_context/bangladesh_tomato_freeze.json) ·
[Independently reconciled metrics and examples](potato_context/bangladesh_tomato_metrics.json) ·
[Prediction records](potato_context/bangladesh_tomato_predictions.csv)

The recipes for restoring the source are in the [data guide](../data/README.md).
From the project root, with the original/default paths and checkpoints in
place, run the following in a new output directory for verification of the
same frozen protocol. Reuse the saved prediction records when images are
unavailable. Keep these images and scores out of training and selection.

```sh
.venv/bin/python -m leafcheck.tomato_holdout freeze --output runs/tomato_transfer_reproduction
.venv/bin/python -m leafcheck.tomato_holdout evaluate --output runs/tomato_transfer_reproduction
```

## PlantSeg provenance and lesion crops

Archive records identified all 7,774 PlantSeg originals inside the inherited
master. The attribution check matched filename, archive size and CRC32, and
the frozen master SHA-256. The frozen split retains 5,301 in training, 1,206
in validation and 642 in primary test. A further 625 were excluded in the
original audit. There are 1,081 official PlantSeg test images in LeafCheck
training. An evaluation on the publisher's full test split would reuse
training images. [Publisher archive](https://zenodo.org/records/17719108) ·
[Data paper](https://www.nature.com/articles/s41597-025-06513-4) ·
[Source reconciliation](potato_context/plantseg_source_audit.json)

The derived release instead uses lesion annotations from compatible
parents assigned to training in both studies. It yields 722 crops over
98 labels. A crop keeps a margin around the annotated region and at least
55% of each original dimension. Visual review found that some photographed
symptoms occur on fruit and stems as well as leaves. The manually inspected
examples retain the marked region. An independent reconstruction produced
the same manifest and release seal. The treatment receives these crops in
extra replay slots. The control receives those exact parent photos in
the same positions. The master data, prior supplementary data, seeds and
optimization settings match. Both arms score ordinary images, using the
same preprocessing without annotations at inference.
[Registered training recipe](../configs/leafcheck_plantseg_regions_v1_training.json)

After restoring the [release](../data/README.md#plantseg-annotation-experiment),
run the command below. Use `--resume` with the same release, configuration,
trainer revision and run directory after an interruption.

```sh
.venv/bin/python -m leafcheck.field_expansion \
  --data data/processed/leafcheck_plantseg_regions_v1 \
  --config configs/leafcheck_plantseg_regions_v1_training.json \
  --run runs/field_v3/plantseg_regions
```

## What the app does

Automatic mode uses the original freeze with the prior uncertainty rule.
Choosing **Potato** is an explicit request to rescale the original CNN's three
potato scores to sum to one. They are conditional model scores, with neither
a diagnostic interpretation nor the automatic mode's decision threshold.
The comparator cards retain their ordinary label sets. Choosing a new photo
resets the selector to Automatic. The app does not infer or correct the user's
manual crop choice.

The extracted demo showed Tomato bacterial spot in automatic mode for a
labeled Okara early blight fixture. Given potato, the same upload displayed
Early blight at 81.8%. A second labeled early blight fixture still displayed
Healthy at 89.9% given potato. Both cases use source labels as references.
These fixtures demonstrate the real upload path, including a remaining
high-score mistake. A deliberately invalid JPEG showed a readable error.
The [browser record](potato_context/demo_browser.json) and
[archive verification](potato_context/demo_archive_verification.json)
give the exact checkpoint, ZIP, fixtures, API agreement and HTTP timing.

Start the app in the repository with the command below. On another supported
Apple silicon Mac, extract `artifacts/LeafCheck-potato-context-demo.zip`
and use the app-local requirements and launcher in its README. The ZIP
contains the weights and runs without research images or notebooks.

```sh
.venv/bin/python leafcheck-demo/backend/server.py --port 8766
```

## Reproduction

The research commands assume access to the frozen master, the cached training
statistics at 128 and 224 pixels, the saved base checkpoint and seed-435
checkpoint, and the exact prior field releases documented in the
[data guide](../data/README.md). No original test images need to be reevaluated
to rebuild the app or to review this study.

```sh
.venv/bin/python scripts/prepare_okara_holdout.py fetch
.venv/bin/python scripts/prepare_okara_holdout.py build
.venv/bin/python scripts/prepare_okara_holdout.py verify
.venv/bin/python -m leafcheck.potato_study --output runs/field_v3/potato_pair
.venv/bin/python -m leafcheck.potato_policy screen --output runs/field_v3/policy \
  --training runs/field_v3/potato_pair
.venv/bin/python -m leafcheck.potato_policy freeze \
  --study runs/field_v3/policy --output runs/field_v3/frozen_okara
.venv/bin/python -m leafcheck.potato_policy evaluate --output runs/field_v3/frozen_okara
```

Build commands expect an absent output directory. Check input provenance before
running a reproduction. Pass `--resume` with the same directory and unchanged
configuration to continue an interrupted scratch run. Reuse the published
Okara scores when auditing completed work. Do not use its errors or scores
to choose new weights, crop thresholds or data.
