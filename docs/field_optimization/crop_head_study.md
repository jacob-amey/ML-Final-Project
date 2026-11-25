# Crop classifier study

The [registered screen](../../configs/leafcheck_crop_head_v1.json) tested
whether a separate crop classifier could improve automatic crop and
condition predictions on the existing grouped field development sources.
The 177-label course checkpoint supplied each crop's condition scores.
The two systems' crop probabilities were averaged with equal weight,
and the condition scores retained their relative ranking inside each crop.

## Matched development

Both arms used a randomly initialized four-block PyTorch CNN with
43 crop outputs and 128-pixel input. For each seed, training used a
sample of 13,194 master photos and 4,082 additional occurrences per
epoch. The control used master replay and the treatment used screened
supplemental photos, including 529 from the Ghana CCMT crop-only
release. All four runs completed six epochs. The documents and
[data guide](../../data/README.md#ccmt-crop-only-training-sample) describe
the broad publisher labels, exact image hashes and source grouping.

| System | Master joint F1 | Master correct / 16,836 | Mean field joint F1 |
| --- | ---: | ---: | ---: |
| Original 177-label CNN | 0.649221 | 15,111 | 0.050016 |
| Scratch control, seed 434 | 0.645696 | 15,074 | 0.051546 |
| Scratch field arm, seed 434 | 0.646089 | 15,100 | 0.091941 |
| Scratch control, seed 435 | 0.646450 | 15,093 | 0.050995 |
| Scratch field arm, seed 435 | 0.647168 | 15,081 | 0.083875 |

Field macro-F1 averaged the represented conditions within each source
and weighted the grouped Sheikh, Mduma and Pakistan datasets equally.
Both seeds met the registered limits. The main gain was on Mduma
potatoes. Its joint F1 improved by 0.123099 and 0.098900. Sheikh joint
F1 stayed at zero on both seeds, and the Pakistan gain was 0.002678
for both seeds. The study isolates the contribution of the screened
field photo mixture. It does not estimate CCMT's marginal effect.

A NumPy class-name serialization defect prevented the default
weights-only loader from opening the first control checkpoint. The
original files were preserved. Recovery changed the metadata to
ordinary Python strings in a new run directory, verified every weight
tensor against the original, and resumed without repeating the six
completed epochs. The selected checkpoints from both arms and seeds
reloaded and reproduced their exact field confusion counts.
[Paired decision and recovery hashes](crop_head_matched.json)

## Independent confirmation

The declared rule selected seed 434 epoch 1 before confirmation
inference. Both systems emitted predictions over all 177 labels.
Holeta gives a separately captured potato source with healthy and late
blight labels. TLID gives a Brazilian greenhouse source with a
compatible healthy and bacterial spot subset. It has no tomato
early- or late-blight label. Each source stayed a whole test-only
group. Spot checks showed mixed in-plant and detached potato views,
and detached tomato leaves on plain backgrounds. This check measures
transfer to these collections. Its coverage is narrower than informal
garden uploads. [Holeta](https://data.mendeley.com/datasets/v4w72bsts5/1) ·
[TLID](https://data.mendeley.com/datasets/kt64b2kh89/1)

| Source | Original correct | Candidate correct | Original → candidate macro-F1 | Original → candidate wrong crop |
| --- | ---: | ---: | ---: | ---: |
| Holeta potato | 6 / 125 | 6 / 125 | 0.089402 → 0.088455 | 118 → 112 |
| TLID tomato | 3 / 120 | 4 / 120 | 0.046875 → 0.061538 | 113 → 108 |

Mean joint macro-F1 improved by 0.006858 against the +0.020
requirement. The wrong-crop rate fell by 4.48 percentage points against
the 5.00-point requirement. Neither individual condition lost recall.
The candidate failed the predeclared gate and the app kept its frozen
177-label checkpoint. The 245 images and scores are reserved as an
opened evaluation and excluded from subsequent development.
[Freeze](crop_head_confirmation_freeze.json) · [Metrics](crop_head_confirmation.json) ·
[Predictions](crop_head_predictions.csv)

## Reproduction

```sh
.venv/bin/python scripts/prepare_ccmt_crop_training.py fetch
.venv/bin/python scripts/prepare_ccmt_crop_training.py build
.venv/bin/python -m leafcheck.crop_head_study --seed 434 --output runs/field_v6/crop-434-new
.venv/bin/python -m leafcheck.crop_head_study --seed 435 --output runs/field_v6/crop-435-new
.venv/bin/python scripts/prepare_crop_confirmation.py fetch
.venv/bin/python scripts/prepare_crop_confirmation.py build
```

For interrupted training, pass `--resume` with the run's same seed
and directory. Use the saved prediction rows and test membership to
recompute the historical confirmation. Fresh experiments require new
independent confirmation groups.
