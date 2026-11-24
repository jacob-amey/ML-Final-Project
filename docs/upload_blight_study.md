# LeafCheck upload and blight study

**Decision.** Retain the frozen 177-label custom CNN. Five specialist runs and
two alternate framing policies did not earn promotion. The live upload handler
now gives an accurate size-limit response for an oversized JPEG. The original
and corrected handlers gave exactly the same 177 scores on the five supported
diagnostic files.

## Scope and verification

Jacob Amey and Demetre’s CSC-427 project classifies a single photo into one
inherited plant-condition label. The custom PyTorch CNN starts with random
weights. *AI Project Build Guide.docx*, steps 1 and 3–5, describes defining the
prediction and expected answer, recording sources, grouping related examples,
and establishing a frequency rule. Steps 7–9 call for validation-led revision,
frozen testing, and a fresh upload demo. The available *Final Project
Requirement_Fall2026.docx* is headed CSC 434. Its Results Validation/Testing
section calls for comparative performance evaluation and software testing.
Exact applicability to CSC-427 is provisional. The from-scratch requirement
comes from Jacob’s instruction.

The starting checkout was clean on `dev` at `263b64a`. The launch port had
no owner, and process checks found no trainer. Apple MPS was available. The
frozen release, archives and teammate’s interface files were preserved.
This study uses the group-assigned train and validation membership of the
frozen release; the reserved test and old PlantDoc evaluation did not
inform changes. The previously frozen scores are included for context after
selection. Existing software regression tests also exercise the original
packaged examples without scoring any candidate on those examples.

The normal launcher runs `leafcheck-demo/backend/server.py`. The browser
sends the original file to `POST /api/predict`. The handler validates the
format and size, selects frame zero, applies EXIF orientation, converts to RGB,
and writes a temporary lossless PNG. The explicitly referenced final model
loads the PNG through the same 224 × 224 aspect-preserving, bilinear gray
letterbox and training-only RGB normalization as standalone evaluation.
The model has 724,713 parameters and uses eight-view mean softmax scoring.
The UI splits the checkpoint's ordered `crop__condition` label and displays
the three largest scores. If the frozen artifact is missing or fails checks,
the final card reports an error and the other independently loaded cards can
still run. The final card does not fall back to a pilot. An explicit environment
override can select a different artifact, which is visible through
`/api/status`. No override was set in verification.

The runtime selects `leafcheck-demo/models/final177/3fa7e014945736b0/model.pt`
through `leafcheck-demo/models/final_model.json`. Its research copy and class
map live in `artifacts/leafcheck177-v1/`.

| Identity | SHA-256 |
| --- | --- |
| Selected checkpoint | `3fa7e014945736b036f85aab1d0af06eafa27a7193a406c9c6349b4a133f23a8` |
| Ordered class map | `53534f8af1d4fda317df463c3dfe160d488dec15e9c16e52a81352bf86e4b03a` |
| Inference preprocessing | `e5f90e4252b3ee708a9fa0385346429b9bed758d3ad36e5ec6fc0fd3abeaade9` |
| Frozen data release | `4287764ba029d96abaa0532d7e1d14d2656cadc774f445be7785b03b69d2903c` |
| Blight supplement | `c2147459192fcc2d49d7787162960786353302220274719d512dedb76ea7cb78` |

## Why the predictions failed

**Confirmed on substitutes.** The app reproduced errors already present in
the CNN's standalone scores. The upload trace captured the temporary PNG from
the actual handler. Five supported originals had identical EXIF-oriented RGB
pixels, resized RGB bytes and normalized tensors across the two paths. Maximum
absolute differences were zero for raw logits, eight-view policy logits and
all 177 ordered scores. A browser file-chooser replay verified the displayed
crop, condition and rounded score. EXIF rotation, grayscale and RGBA PNG, and
first-frame MPO are covered by the existing integration checks. Jacob’s own
files were unavailable in the repository and session attachments. Their
diagnoses and the specific causes of their predictions remain unverified.

| Diagnostic original | Reference annotation | Before and after in the app |
| --- | --- | --- |
| Master validation `tomato__late_blight__000001.jpg` | Tomato late blight | Tomato late blight, 99.5% |
| Master validation `tomato__late_blight__000039.jpg` | Tomato late blight | Tomato early blight, 66.1% |
| Master validation `tomato__early_blight__000386.jpg` | Tomato early blight | Tomato late blight, 36.6% |
| Bugwood `1568028.jpg`, broad view with multiple leaves and fruit | Tomato late blight | Zucchini bacterial wilt, 59.0% |
| Sheikh validation PNG `a4cc9c83…` | Tomato early blight | Rice leaf blast, 94.8%, verified in API |
| Banu `IMG_6758.jpg`, 4284 × 5712 | Tomato late blight | Rejected both times. API response corrected from generic HTTP 400 to HTTP 413 with the existing 24-megapixel limit. |

The Bugwood photo is attributed to Edward Sikora, Auburn University,
Bugwood.org, CC BY 3.0. [Catalog source](https://www.ipmimages.org/browse/image/1568028)

**Best supported explanation.** The model scored 7,385/7,586 on master
validation photos with a documented PlantVillage leaf ID, 20/64 on documented
PlantDoc byte matches, and 0/272 on the screened new non-Bugwood supplement.
These slices differ in labels as well as sources. The four blight labels also
illustrate the shift directly. See the source-specific counts below. Visually
reviewed files range from a closely framed, evenly positioned single leaf on
gray to a small leaf on A4 paper and dense outdoor scenes with foliage, fruit,
partial leaves and uneven light. Fewer pixels on the lesion and correlations
with background are plausible mechanisms. The evidence does not isolate one
causal visual feature.

**Class imbalance is a weaker explanation for this failure pattern.** Tomato
late blight had 1,450 training files in 959 recorded groups, versus 817 files
in 328 groups for tomato early blight. Potato early and late blight had
824 files/338 groups and 835 files/346 groups. The matched weighted run
had slightly lower full-validation F1 than its unweighted counterpart and
lost its single supplemental correct prediction. Group support alone also
does not predict recall. Squash powdery mildew had 108 training groups and
10/31 validation recall. Potato healthy had 28 training groups and 27/28
recall. These are observational comparisons with different visual difficulty
and label support.

## Directional blight errors

The left directional column uses every labeled late-blight photo for the
crop as denominator. The reverse uses every early-blight photo. Wrong-crop
errors are separate and have the same source-specific denominators. Celery
has an early-blight label only. A dash indicates an unavailable class or no
examples in the slice.

| Development slice | Crop | Late → early | Early → late | Wrong crop from late | Wrong crop from early |
| --- | --- | ---: | ---: | ---: | ---: |
| Master validation | Tomato | 17/414 | 3/233 | 25/414 | 1/233 |
| Master validation | Potato | 0/238 | 0/236 | 7/238 | 0/236 |
| Master validation | Celery | — | — | — | 3/7 |
| Master documented PlantVillage | Tomato | 17/348 | 3/232 | 23/348 | 0/232 |
| Master documented PlantVillage | Potato | 0/236 | 0/236 | 5/236 | 0/236 |
| Sheikh paper photos | Tomato | — | 0/56 | — | 56/56 |
| Banu collection | Tomato | 0/12 | — | 12/12 | — |
| Banu collection | Potato | 0/18 | — | 18/18 | — |
| Bugwood field fixtures | Tomato | 0/5 | 0/1 | 5/5 | 1/1 |
| Bugwood field fixtures | Potato | 0/2 | — | 1/2 | — |

For master validation, same-crop wrong-condition errors were 19/414 for
tomato late and 8/233 for tomato early. Late tomato was called potato late on
21/414, the largest confusion for that class. The largest overall directed
errors included rice leaf blast → rice healthy (49), bell pepper bacterial
spot → tomato Septoria leaf spot (41), and soybean *Diabrotica speciosa* →
soybean caterpillar (37). No disease annotation was independently diagnosed by
the project. See [directional counts by source](upload_blight_error_breakdown.csv)
and `runs/upload_blight_v1/baseline.json` for supports and per-class scores.

## Data and candidate decision

The frozen release contains 59,008 training, 16,836 validation and 8,425
reserved test images. It has 36,219, 8,282 and 3,948 recorded groups in those
respective splits. Missing specimen metadata means group counts approximate
dependence. Fifty-three thousand twenty-five images lack documented leaf IDs.
The frozen audit verified exact byte and decoded rotation/reflection hashes,
near-copy candidates, known related views, conflicting labels and the recorded
prior exposure list. It kept 570 recognized preexisting offline transforms
in training only. New training transforms ran after assignment. The original
split is defensible given the recorded evidence, with potential undetected
crops, edits, or related specimens remaining. Measured class, crop, split and
source support is in `runs/upload_blight_v1/support_by_source.csv`. Master
provenance is an evidence-based bucket, and many original source identities
remain unresolved.

The supplement was hashed and screened against the full audited master and
registered PlantDoc images. The 1,146 supplemental training images came from
four recorded collection groups, including 436 tomato late-blight images
from a single Sheikh capture date and 19 grouped potato late-blight images
from the Mduma–Mtali archive. Sheikh and Mduma–Mtali license their releases
under CC BY 4.0. The validation-only Banu collection excludes files flagged
as augmented. The 272-photo development slice consists of 233 Sheikh paper
photos from held-back dates and 39 Banu photos from a different source.
Thirty-seven Banu originals exceed the app's pixel limit. This slice
measures model performance on files decoded directly for evaluation. Among
the 235 originals eligible by size, the retained CNN also had zero correct.
The eight separately credited Bugwood field photos were diagnostic only
and yielded 1/8 correct. The locally available four relevant PlantWild
folders contain 593 images and all 593 byte hashes were present in the
master records. Renaming their split would not create an independent set.
[Supplement provenance, licenses and method](blight_dataset_v1.md)

The full master validation score is the mean of the 177 class F1 scores.
A class with zero precision or recall contributes zero, including a class
with a zero denominator. On the supplement, the mean covers the four
represented labels while all 177 predictions are eligible. The eight field
photos have three represented labels and were excluded from promotion.
Compare all candidates on identical image membership. The operational gain
floors were +0.001 full validation macro-F1 and +0.025 supplement macro-F1.
These thresholds express project choices. The scores are paired by image,
and related views are not independent. No new confidence threshold was
deployed.

| System, all 177 outputs eligible | Master val. F1 | Master correct / 16,836 | Supplement F1 | Supplement correct / 272 | Field correct / 8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original deployed checkpoint | 0.649221 | 15,111 (89.75%) | 0.000000 | 0 | 1 |
| Selected checkpoint with API message fix | 0.649221 | 15,111 (89.75%) | 0.000000 | 0 | 1 |
| Six-way specialist, master only, unweighted | 0.649675 | 15,116 (89.78%) | 0.000000 | 0 | 1 |
| Six-way specialist, expanded, unweighted | 0.649727 | 15,116 (89.78%) | 0.002674 | 1 | 1 |
| Six-way specialist, expanded, weighted | 0.649583 | 15,112 (89.76%) | 0.000000 | 0 | 1 |
| Seven-way routing with other-label negatives, master only | 0.649314 | 15,111 (89.75%) | 0.000000 | 0 | 1 |
| Seven-way routing with other-label negatives, expanded | 0.649221 | 15,111 (89.75%) | 0.023077 | 9 | 1 |
| Clearly isolated specimen crop, original weights | 0.643849 | 15,019 (89.21%) | 0.119344 | 13 | 1 |
| Average original and isolated specimen crop scores | 0.648679 | 15,101 (89.69%) | 0.000000 | 0 | 1 |

The first specialist was limited to six tomato/potato labels and gated by
the incumbent’s actual top prediction. This controlled unweighted and
weighting comparisons but could not reach most wrong-crop photos. The second
specialist was trained with 7,151 group-sampled negatives and could route
to six labels from any incumbent crop. It used its own predicted label.
It rescued nine supplement photos without raising master validation F1.
The unweighted expanded six-way candidate’s +0.000506 master gain did not
clear the floor and corrected one supplemental file. Cropping 455 eligible
files improved some isolated-paper predictions, but cost 92 correct master
predictions, added 63 wrong-crop master errors and did not help the field
fixtures. All five specialists used random initialization and completed
their declared budgets or plateau stops. Saved checkpoint reload reproduced
their selection scores. Fitting longer to the same capture groups or
adjusting routes against eight correlated field photos lacks a sound basis.
A more meaningful follow-up needs independently labeled garden photos from
new plants, collections and photographers that can be grouped into genuinely
independent development and test sets. That data is the main remaining limit.

For the retained model, the previously frozen result remains **7,556/8,425
(89.69%)**, macro-F1 **0.616940**. The separately registered historical
PlantDoc slice remains **26/183 (14.21%)**, three-label macro-F1 **0.219846**.
Neither is a new field result. No candidate was scored on either holdout,
and no new independent field holdout was available. The prior simple rule
was 731/8,425. More detail in [the experiment record](upload_blight_experiments.csv)
and `runs/upload_blight_v1/per_class.csv`.

## Reproduce and run

Run from the repo root using the locked research environment and exact
dataset paths. The master and supplement manifests are
`data/processed/leafcheck177_final/manifest.csv` and
`data/processed/leafcheck_blight_v1/manifest.csv`. Preparation and restoration
are described in [the data guide](../data/README.md). Full master validation
argmax records come from the earlier frozen inference comparison at
`runs/final177/inference_comparison/residual177_224_seed434-d4_probability_mean-predictions.csv`.
The combined course ZIP also contains that CSV.

```sh
.venv/bin/python scripts/prepare_blight_dataset.py verify
.venv/bin/python -m leafcheck.upload_study --output runs/upload_reproduce_v1
.venv/bin/python -m leafcheck.upload_study --output runs/upload_reproduce_v1 --train original_unweighted --seed 434
.venv/bin/python -m leafcheck.upload_study --output runs/upload_reproduce_v1 --train expanded_unweighted --seed 434
.venv/bin/python -m leafcheck.upload_study --output runs/upload_reproduce_v1 --train expanded_weighted --seed 434
.venv/bin/python -m leafcheck.upload_router_study --input runs/upload_reproduce_v1 --seed 434
.venv/bin/python -m leafcheck.upload_router_study --input runs/upload_reproduce_v1 --seed 434 --master-only
.venv/bin/python -m leafcheck.upload_crop_study --input runs/upload_reproduce_v1 --device cpu
.venv/bin/python scripts/verify_saved_evaluation.py --release data/processed/leafcheck177_final
```

Append `--resume` to the exact command for an interrupted specialist using
the same run path and seed. Its `last.pt` contains the last completed epoch,
model, optimizer, scheduler and RNG state. The completed experimental runs
above require a new output directory to start again. For the original training
recipe, the pinned configuration is `configs/residual177_224_seed434.json`.
Run `.venv/bin/python -m leafcheck.data --size 224 --workers 6` to prepare the
original training/validation cache. For an interrupted reproduction at
`runs/residual177_224_reproduction`, resume with
`.venv/bin/python -m leafcheck.train --run runs/residual177_224_reproduction --resume`.
For a fresh reproduction, pass the same `--run` with
`--config configs/residual177_224_seed434.json`.

Launch the repository app with `./leafcheck-demo/Launch\ LeafCheck.command --port 8765`.
For the trace, start the launcher before running:

```sh
.venv/bin/python scripts/trace_upload.py --url http://127.0.0.1:8765 \
  --cases data/manifests/leafcheck_upload_diagnostics_v1.json \
  --output runs/upload_reproduce_v1/api_trace.json
LEAFCHECK_TEST_URL=http://127.0.0.1:8765 .venv/bin/python -m unittest discover -s leafcheck-demo/tests -v
```

The current checkout passed 16 website tests and nine research pipeline
tests, and the browser observations are in
[the verification receipt](upload_blight_verification.json). The course ZIP
and demo ZIP from the earlier delivery remain historical. The new upload
response is in the repository app source.
