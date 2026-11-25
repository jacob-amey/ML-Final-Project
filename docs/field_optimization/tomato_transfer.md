# Tomato transfer study

The course model remains `residual177_224_seed434` at epoch 53 with
`d4_probability_mean`. This study investigates two reported upload symptoms.
The team observed unrelated crops and possible early-versus-late blight
errors. The study uses photos with inherited source labels. The team's
original photos and reference diagnoses have not been independently verified.

## Matched development results

Both arms started from random weights with the same compact PyTorch CNN
and ordered ten tomato labels. The control saw 11,828 master training
images plus 1,874 same-label master replay draws per epoch. The treatment
saw the same 11,828 images plus 1,874 screened training-only field images.
Control seeds 434 and 435 stopped at epochs 9 and 11. The treatment
completed 12 for both seeds. The sampler matched the label dose each epoch.

| Method with tomato supplied | Grouped source mean development macro-F1 | Tomato-only master validation macro-F1 |
| --- | ---: | ---: |
| Original CNN | 0.187248 | 0.975798 |
| Scratch control, seeds 434 and 435 | 0.279810 | 0.710832 / 0.749742 |
| Scratch field training, seeds 434 and 435 | 0.315641 | 0.849532 / 0.822145 |
| Original and field 50:50 blend, seeds 434 and 435 | 0.317404 | 0.967749 / 0.966277 |

The macro-F1 mean gives equal weight to the 233-image Sheikh and
142-image Pakistan tomato development sets. Banu remained diagnostic.
The standalone treatment missed the master guard on both seeds. Routing
the treatment by the original predicted crop reduced joint field macro-F1
by 0.016558 on each seed. The route applied to only 4 Sheikh and 28
Pakistan photos. The known-tomato blend passed its development criteria
against both the original and the same-fraction matched control
(0.257144 mean field macro-F1). Fractions 0.125 and 0.25 missed the
criteria. Seed 434 epoch 11 was selected before PlantCity scoring.

## Frozen independent PlantCity check

The screened release contains 299 original photos, six crops and 13
represented labels. The tomato-only comparison contains the same 134
photos and six represented labels for both systems. All ten tomato
outputs were eligible in that comparison. Automatic inference allowed
all 177 outputs. The data publisher describes real captures in Pakistan.
Some samples show detached leaves on a simple background, and individual
specimen IDs are unavailable. [Publisher dataset](https://data.mendeley.com/datasets/w8kh2xkspx/4)

| Policy | Correct / total | Macro-F1 | Wrong crop |
| --- | ---: | ---: | ---: |
| Original automatic | 26 / 299 | 0.114388 | 224 / 299 |
| Original, tomato supplied | 44 / 134 | 0.298658 | Crop supplied |
| 50:50 blend, tomato supplied | 34 / 134 | 0.233403 | Crop supplied |

| Policy | Tomato early → late | Tomato late → early | Wrong crop on early | Wrong crop on late |
| --- | ---: | ---: | ---: | ---: |
| Original automatic | 0 / 22 | 1 / 21 | 18 / 22 | 14 / 21 |
| Original, tomato supplied | 4 / 22 | 10 / 21 | Crop supplied | Crop supplied |
| Blend, tomato supplied | 3 / 22 | 9 / 21 | Crop supplied | Crop supplied |

The blend lost 0.065255 macro-F1 and 0.074627 accuracy. It lost
0.318182 early blight recall and 0.142857 late blight recall. Both
systems produced correct and incorrect labels in the early and late blight
slices. The directional swap counts alone would miss several other errors.
No potato, celery or other crop contributes an early/late pair to this
release. The original course checkpoint and deployed optional potato
context remain in service. The opened PlantCity images and outcomes
are excluded from candidate design, training and selection.

The [run evidence](tomato_matched.json), [blend validation](tomato_blend_development.json),
[confirmation freeze](plantcity_freeze.json), [metrics](plantcity_confirmation.json),
and [exact predictions](plantcity_predictions.csv) record the outcome.
Each chosen training checkpoint reloaded and reproduced its recorded
field confusion counts. A post-training batch-indexing correction reran
only the automatic scorer from the existing saved checkpoints.

## Reproduction

```sh
.venv/bin/python -m leafcheck.tomato_study --output runs/field_v5/tomato_specialist
.venv/bin/python -m leafcheck.tomato_condition_blend --study runs/field_v5/tomato_specialist --output runs/field_v5/tomato_condition_blend.json
.venv/bin/python scripts/prepare_plantcity_holdout.py fetch
.venv/bin/python scripts/prepare_plantcity_holdout.py build
```

For an interrupted training run, add `--resume` with its unchanged
configuration and run directory. For a completed conditional comparison
with missing automatic scoring, use `--joint-only`. Use the saved
PlantCity predictions and membership to recompute metrics. The exact
original freeze identifies the evaluated checkpoint and policy.

Build the editable field study with a Python environment that has
`python-docx` and the local frozen artifact.

```sh
python scripts/build_submission_report.py --artifact artifacts/leafcheck177-v1 \
  --include-tomato-transfer --output artifacts/submission/LeafCheck_Report_Field_Study.docx
```
