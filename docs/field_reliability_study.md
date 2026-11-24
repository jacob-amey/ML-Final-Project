# LeafCheck field reliability follow-up

**Selected system.** Keep the from-scratch 224-pixel residual CNN and all 177
labels. Add a validation-selected rule to the application that withholds the
visible final-model suggestion when its highest score is below
**0.5857925415039062**. The raw rankings and historical results are available
for comparison. The score is uncalibrated.

The new field collection had 218 images. The rule reduced incorrect visible
suggestions from **209 to 30** by withholding **185** results, including **6**
of the 9 previously correct predictions. Only **3 of the 33** suggestions
shown were correct. This is an improvement in how the app handles uncertainty.
It provides no evidence of ready-to-use field diagnosis.

## What was audited

The assignment requires a custom CNN trained from random weights, a defensible
comparison, and an interactive local demo. The original submission keeps the
177-way network and its original test data. The field follow-up is separate.
See *AI Project Build Guide.docx*, steps 1, 3–5 and 7–9, and the
[course scope reconciliation](scope_reconciliation.md). The CSC 434 heading on
the available rubric still makes exact CSC-427 applicability provisional.

The repository was clean at `c2cd005`. No trainer held the accelerator.
Apple MPS extracted the frozen features, and CPU training was sufficient
for the classifier heads. The original architecture, checkpoint, dataset
and holdout remained unchanged. The chosen model is the originally trained
724,713-parameter CNN. The head experiments warm-started its own linear layer
and fit only that layer. They used no externally pretrained weights. Shared
inference uses frame zero, EXIF correction, RGB, bilinear letterboxing, and
training-only normalization, then averages scores from eight orientations.
The new gate is a constant comparison on the returned score and adds no
network forward pass.

The frozen master keeps 59,008 training, 16,836 validation, and 8,425 test
images. The earlier audit recorded 48,449 groups and 53,025 images without
documented leaf IDs. It screened exact, rotated, reflected and approximate
duplicates, assigned known siblings together, and kept 570 recognized
offline augmented files in training only. Image provenance and true
specimen independence remain incomplete. The [full dataset audit](final_dataset_audit.md)
and [upload trace](upload_blight_study.md) document ambiguity, prior exposure,
and pixel/logit/API parity for the original misclassification examples.
Jacob’s original files are still unavailable.

| Frozen master metric | Validation | Historical test |
| --- | ---: | ---: |
| Top-1 accuracy | 89.75% | 89.69% |
| Macro-F1 | 0.649221 | 0.616940 |
| Balanced accuracy, mean recall | 0.647981 | 0.619776 |
| Weighted F1 | 0.895327 | 0.894153 |
| Macro precision | 0.666045 | 0.633107 |
| Classes with zero F1 | 11 | 23 |

Validation top-3 accuracy was 96.15%. In 143 validation errors, the highest
score was at least 0.90. Median highest scores were 0.998 on correct
predictions and 0.527 on incorrect ones. The [validation class audit](field_reliability/per_class_val.csv)
and [historical test class audit](field_reliability/per_class_test.csv)
include training group counts, support, precision, recall, F1, and one-vs-rest
false-positive rates, false-negative rates and accuracy. One-vs-rest accuracy
can be high for small classes because of the large number of true negatives.
The lowest observed F1 is zero on both splits. These historical test
predictions were opened only in the earlier study and were kept out of
follow-up fitting and selection.

Support alone did not explain the failures. In prior same-source tomato
validation the final CNN confused late and early blight in both directions.
Tomato late blight has more training data than tomato early blight. In the
new field test, **195/218** answers named a different crop. Only **1/82**
late-blight photos were called potato early blight and **5/62** early-blight
photos were called potato late blight. Source dependence and wrong-crop errors
are a higher priority than tweaking early versus late loss weights. Visual
review showed differences including busy scenes, hands, view distance and
hard black borders in the source images. The review cannot isolate a
causal feature or independently confirm the publisher’s labels.

## New held-out field collection

PLDD-UP v1 is CC BY 4.0 and attributed to Prakash Kumar Singh, Arun Yadav,
Divakar Yadav, Sarthak Tiwari and Aseem Chandel. Its publishers describe potato
images collected in fields in Uttar Pradesh. The archive groups are healthy,
early blight and late blight. [Publisher record](https://data.mendeley.com/datasets/3j4nfkvp2n/1)

A fixed filename-hash rule chose 96 candidate files from each public ZIP.
The builder requested ZIP byte ranges and verified each member against the
archive directory’s size and CRC. Publisher whole-archive SHA-256 values are
recorded. The sampled build did not fetch or hash the complete source ZIPs.
Candidate image bytes are hashed individually. Exact and approximate overlap
screening used 90,574 master audit entries, the earlier 183-image PlantDoc
slice and all 1,426 blight supplement files. Same-label repeat views were
collapsed, and oversized/unsupported files were excluded for app compatibility.
After screening, 62 early-blight, 74 healthy and 82 late-blight photos remained.
All stayed together as a single-source external test. Unknown related plants,
mislabeling and edited-image copies can remain. The test represents this
collection, not a random sample of all gardens or devices.

The available PlantWild tomato and potato folders overlap the master. An
additional Banu–Deb potato archive was considered and kept out of this
study. Only 36 files in the two compatible folders were marked original, and
the source shares a contributor with existing supplemental data. The
broader VegHealth14 release is a merge of existing public sources, making
source independence difficult to establish. More coverage should come
from separately documented collections, rather than mixing in unverified
repackages. [Banu–Deb record](https://data.mendeley.com/datasets/d5b3fzpw3g/1),
[VegHealth14 provenance](https://data.mendeley.com/datasets/c93cpkfb5p/1)

The scope, checkpoint hash, class map, preprocessing, rule, threshold and
field manifest were [frozen before scoring](field_reliability/freeze.json).
A separate rebuild from pinned inputs reproduced the field release manifest.
This reflects the same broader generalization concern observed in the
original PlantVillage paper. [Mohanty, Hughes and Salathé](https://doi.org/10.3389/fpls.2016.01419)

## Experiment decisions

Seven head refits and three probability blends retained the full 177-way
prediction space. They used the same grouped master validation and the
272-image supplement validation. Source groups remained intact. Training
and selection did not open the new field images for scoring. Full master
macro-F1 was primary. The declared head promotion rule required a +0.003
master macro-F1 gain, at most a 0.25 percentage point accuracy drop and
a +0.025 supplement F1 gain.

| Head or signal | Master macro-F1 | Supplement correct / 272 | Outcome |
| --- | ---: | ---: | --- |
| Original | 0.649221 | 0 | Retain weights |
| Master-only refit | 0.641226 | 0 | Reject |
| Expanded ordinary cross-entropy | 0.644920 | 179 | Broad validation regressed |
| Expanded group weights | 0.645438 | 104 | Weaker source gain |
| Expanded crop loss, seed 434 | 0.644906 | 179 | No practical gain over expanded CE |
| Expanded crop loss, seed 435 | 0.639493 | 181 | Broad regression repeated |
| Expanded softened prior, 0.25 | 0.646984 | 177 | Broad validation still lower |
| Expanded softened prior, 0.50 | 0.647786 | 125 | Less source recovery |
| Best full-master blend, half adapted | 0.648460 | 130 | Missed promotion rule |

The refits show that the existing features carry information the original
head failed to use on those supplemental sources. They also show a tradeoff
with broad validation. The expanded CE head produced 67 wrong crops on
the supplement, down from 268, but 1,012 on master validation, up from 956.
The source-specific tradeoff persisted across the interventions.
Balanced-softmax-inspired correction was tested only
as a training loss term, with strengths declared before the runs.
[Decoupled classifier training](https://arxiv.org/abs/1910.09217),
[Balanced Softmax](https://proceedings.neurips.cc/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html)

A grouped split of the master validation set supplied 6,812 threshold-fit
photos and 10,024 check photos held apart from fitting the threshold.
A single temperature fitted on calibration increased check-set 15-bin
ECE from 0.0133 to 0.0253. It was rejected for display. A group-trained
nearest-centroid score, its combination with maximum score, and the
top-two score margin also failed deployment criteria or performed worse.
Maximum score kept 9,014/10,024 master check suggestions (89.92% coverage)
and improved correctness among suggestions from 89.62% to 95.63%. It
withheld 128/272 supplemental predictions, all of which would have been
incorrect. The 144 suggestions it kept on that validation slice were also
wrong. The screen is selective prediction, not detection of all invalid
images. [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html),
[Maximum softmax baseline](https://arxiv.org/abs/1610.02136)

## Field results after the freeze

| Metric on the same 218 originals | Original display | With screen |
| --- | ---: | ---: |
| Suggestions shown | 218 | 33 |
| Correct suggestions | 9 | 3 |
| Incorrect suggestions | 209 | 30 |
| Results withheld | 0 | 185 |
| Correctness among shown | 4.13% | 9.09% |
| Wrong suggestions per upload | 95.87% | 13.76% |

Before screening the field CNN scored 4.13% top-1, 15.14% top-3,
0.072886 macro-F1 and 0.041831 balanced accuracy across the three
represented labels, with all 177 outputs eligible. Its 15-bin ECE was
0.305971 on this one collection. It showed 0/74 correct healthy examples.
After screening all three correct suggestions were early blight. Two of
four unrelated scikit-image demo photos still earned suggestions in a
software smoke check. A picture of coins was assigned to squash powdery
mildew with a 0.9991 score. Example data of this size do not establish an
unrelated-photo detection rate. [scikit-image sample sources](https://scikit-image.org/docs/stable/api/skimage.data.html)

[Field confusion matrix](field_reliability/field_confusions.png) and
[suggestion counts](field_reliability/field_suggestions.png) are based on
the locked predictions. No threshold was changed after opening this set.
More training against the same small number of paper-photo capture dates
is unlikely to resolve the field finding. A meaningful next experiment
would use multiple sources of field photographs with independently
checked labels, plant/session IDs, and a new unexposed collection for
evaluation. The published field set used here can serve as a documented
stress case for future comparison. Keep it separate from new evaluation
claims and threshold decisions.

## Reproduce or resume

Use the pinned Python environment and [the frozen data guide](../data/README.md)
for the original dataset. The original production model remains
`artifacts/leafcheck177-v1/model.pt`, SHA-256
`3fa7e014945736b036f85aab1d0af06eafa27a7193a406c9c6349b4a133f23a8`.
Its maps and preprocessing are pinned by the field freeze. The app points to
the byte-identical model via `leafcheck-demo/models/final_model.json` and
loads `leafcheck-demo/backend/reliability_policy.json` only for that compatible
artifact.

```sh
.venv/bin/python scripts/prepare_field_holdout.py
.venv/bin/python -m leafcheck.refit_study cache --output runs/refit_reproduction
.venv/bin/python -m leafcheck.refit_study fit --output runs/refit_reproduction --recipe expanded_ce --seed 434
.venv/bin/python -m leafcheck.refit_study fit --output runs/refit_reproduction --config configs/leafcheck_refit_prior_v2.json --recipe expanded_prior_0.25 --seed 434
.venv/bin/python -m leafcheck.reliability_study --output runs/refit_reproduction
.venv/bin/python -m leafcheck.select_followup --run runs/refit_reproduction
```

Use the config and seed in the [leaderboard](field_reliability/experiments.csv).
The original recipes are in `configs/leafcheck_refit_v1.json`, and the prior
study is in `configs/leafcheck_refit_prior_v2.json`. The `last.pt` for each
recipe saves the last completed epoch, head, optimizer and CPU RNG state.
For an interrupted run repeat the same `fit` invocation with `--resume`.
Start a fresh output path for a completed run or field dataset rebuild.
The primary CNN training and resume instructions remain in the root README.

To replay the frozen field evaluation, preserve existing result files and
use the recorded decision files in a new directory. This replays inference
on the same already opened field set.

```sh
mkdir -p runs/field_replay
cp docs/field_reliability/freeze.json runs/field_replay/field_freeze.json
cp docs/field_reliability/followup_decision.json runs/field_replay/
.venv/bin/python -m leafcheck.evaluate_field_screen --run runs/field_replay
./leafcheck-demo/Launch\ LeafCheck.command --port 8765
```

The current repository app passed 17 website and nine pipeline tests.
Browser uploads showed an actual withheld result and a correct potato
early-blight suggestion. The API also reproduced an incorrect suggestion
against a source label and retained all ordered scores. See
[verification](field_reliability/app_verification.json). The earlier course
ZIP remains the archived submission. The updated app and field appendices
are a follow-up with their own verification and can be shared locally.
