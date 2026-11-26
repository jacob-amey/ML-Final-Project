# Vineyard and potato field data intake

The field intake checks photo provenance, exact bytes, label compatibility and overlap before assigning a split. The documented course holdouts and historical confirmation sets stay outside fitting and selection.

## GVLiD

The pinned [Mendeley release](https://data.mendeley.com/datasets/wkymf8bhcg/5) advertised four grape conditions, and the [data descriptor](https://doi.org/10.1038/s41597-026-07431-9) described expert labels and vineyard metadata. The downloaded 902,251,054-byte archive matched the publisher SHA-256. All 3,477 JPEG filenames had metadata entries. Folder labels disagreed with their matching metadata labels in 1,344 cases. Two exact decoded groups span conflicting folder labels. The metadata assigns every entry one capture day, 2023-06-15, and a Redmi device. The 1,070 dated EXIF records instead record 2024-10-27, with device fields identifying two camera models across the archive. The evidence does not establish that the advertised vineyard IDs are reliable for split assignment. No photos entered training or evaluation.

The [audit](gvlid_intake_audit.json) records the exact and heuristic comparisons. The original can be restored from the [pinned source](../../data/manifests/leafcheck_gvlid_intake.json). Verify the archive hash, extract with `bsdtar` into the `original` folder alongside the RAR, and run `.venv/bin/python scripts/audit_gvlid_intake.py --output runs/field_v9/gvlid-replay`. Two separate audits reproduced the per-photo table exactly.

## Niphad grape sample

From the [publisher catalog](https://data.mendeley.com/datasets/8nnd2ypcv3/5), 18 healthy and 18 downy mildew images were sampled by a stable hash of each publisher ID. All matched their publisher hashes and decoded. No member of that sample matched under the master and PlantDoc screening method. The reviewed photos show detached leaves on pale backgrounds. They lack EXIF dates in this sample, so the intake leaves this source outside the garden-photo experiment.

## Other potato sources

- The [Sharma and Raheman archive](https://zenodo.org/records/22059910) contains 1,561 field potato JPEGs. The downloaded ZIP matches the publisher's MD5 and local SHA-256 `59e902c87dddf41d1f8dd4ffbfed296d5ab970fac82cdfaffe1e4edfca498554`. The published provenance explicitly excludes per-image conditions, annotations and original split. This archive supports crop-only intake. No photos were admitted for condition learning or scoring.
- The [IIT Mandi sample](https://data.mendeley.com/datasets/pbnw43s6kt/1) has 88 published healthy originals and 217 photos in its blight-affected folder. An initial 18 per label passed byte verification and screening against the master and PlantDoc. The contact sheet shows leaves in place on field plants. The sample of 18 healthy photos shares one capture day and one device. The 18 blight photos span five dates and multiple cameras. These source and condition differences can make camera or session signals correlate with condition. The whole IIT source is reserved for fitting under the new protocol, with validation drawn from separate existing collections. The [study paper](https://doi.org/10.34133/2021/9835724) identifies both early and late blight in the labeled source but the public folder does not separate them per photo. The [registered matched training screen](../../configs/leafcheck_potato_partial_v1.json) therefore preserves the early-or-late label.

Sample IDs, file paths and exact hashes are pinned in the [sample registry](../../data/manifests/leafcheck_field_v9_preflight_files.csv). The [IIT full catalog](../../data/manifests/leafcheck_iit_potato_intake_files.csv) pins all available files in the two folders. Use `.venv/bin/python scripts/audit_field_preflight.py fetch` and then the same command with `audit` to reproduce the sample checks. Add `--iit-potato` to both commands to restore and audit the full IIT catalog with the prior-release protections.

## Matched potato training result

All 305 IIT photos passed byte verification and the overlap screen. All 88 healthy photos share one camera and day. Twenty-eight blight photos lack capture dates. Eighteen filenames say late blight and none say early blight. The experiment retained the publisher's coarse folder labels for all 217 blight photos. The intake screen does not establish one distinct leaf per image. The [audit record](iit_potato_audit.json) preserves observed capture metadata and labels.

The study kept 3,200 master and Mduma training originals in both runs, with 305 additional slots per epoch. The control filled those slots using sampled training photos. Both arms used the same 88 healthy labels and 217 early-or-late targets in these slots. An early-or-late target rewards the sum of those two output probabilities. It supplies no direct supervision about which blight is present. Both custom CNNs started with random initialization, completed 18 epochs, and reloaded the selected epoch-15 checkpoint.

| Development split | Matched control correct | Matched control macro-F1 | New field photos correct | New field photos macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Mduma date-held potato | 191/211 | 0.899476 | 191/211 | 0.900792 |
| Master potato | 480/502 | 0.938809 | 465/502 | 0.881588 |

On Mduma, early blight recall fell from 55/61 to 52/61. Early-to-late errors changed from 6/61 to 7/61. Late blight recall stayed at 97/100 and late-to-early changed from 2/100 to 1/100. Healthy recall rose from 39/50 to 42/50. The +0.001316 field macro-F1 gain missed +0.030, and the master macro-F1 and accuracy losses exceeded their limits. Neither selected specialist met the master's guard against the reference CNN given potato. The study ended without a seed repeat. The demo checkpoint and its optional potato context stay in service.

Reproduce the comparison after restoring the master and field releases and preparing the 128-pixel cache. Use `.venv/bin/python -m leafcheck.potato_partial_study --audit runs/field_v9/iit-potato --output runs/field_v9/potato-partial-replay`. Add `--resume` for an interrupted run with the same identity. See the [full results and fingerprinted checkpoint record](potato_partial_study.json).

## Pretoria corn preflight

The [University of Pretoria release](https://researchdata.up.ac.za/articles/dataset/Diseases_of_maize_in_the_field/20237613)
provides originals and multi-label annotations under CC BY 4.0. The
verified table has 2,355 image records, including 891 with more than one
positive flag. Of the single-label photos, 1,109 map to four existing
LeafCheck labels. The mapping uses the source's no-foliar-symptoms flag
as the healthy proxy.

The [pinned preflight](../../data/manifests/leafcheck_pretoria_corn_intake.json)
used the publisher's archive directory and exact byte ranges. The 16
sample photographs passed the master and PlantDoc overlap screen and span
six EXIF days. The contact sheet shows outdoor backgrounds and a few
images framed by hands or field-study equipment. The filename archive
is flat, so EXIF dates are preliminary capture grouping evidence.
The [receipt](pretoria_preflight.json) and [sample manifest](../../data/manifests/leafcheck_pretoria_preflight.csv)
preserve those observations. No sample entered training or evaluation.
If expanded, screen all protected releases before using the source and
keep repeated views grouped across development splits.

## Expanded corn field validation

The same content-independent ID rule took up to 24 original photos from
each of the four compatible Pretoria labels. All 96 originals decoded,
and two candidate repeated views were excluded. The 94 development
images span nine EXIF days. Common rust appears on two dates, limiting
the number of genuinely separate captures. Keep the entire Pretoria
source in development. The [expanded checklist](pretoria_expanded.json)
records the exact byte and grouping evidence.

The [protocol](../../configs/leafcheck_pretoria_baseline_v1.json) pinned
bytes, class order and the deployed checkpoint before the first score.
Automatic mode retained all 177 outputs. The supplied-corn diagnostic
retained all six existing corn outputs, including labels absent from
these photos.

| Method | Top-1 correct | Macro-F1 across four represented labels | Wrong crop | Right crop, wrong condition |
| --- | ---: | ---: | ---: | ---: |
| Automatic | 21/94 | 0.198062 | 22 | 51 |
| Supplied corn | 26/94 | 0.202778 | 0 | 68 |

Both modes got zero of 24 reference common-rust images and zero of 24
healthy-proxy images right. Each returned the separate inherited
`corn__rust` label for eight common-rust images. Master training contains
897 `corn__common_rust` and 107 `corn__rust` examples. The master preserves
them under different source folder names. The 68 PlantDoc-linked common-rust
training rows and the family labels do not establish how the generic rust
class differs biologically. Keep both frozen labels during this study.
The study shows that supplying a crop alone leaves substantial condition
errors. It provides no estimate of general reliability across all crops.

[Per-class and date-level results](pretoria_baseline.json) ·
[Exact prediction records](pretoria_predictions.csv). Reproduce with
`.venv/bin/python -m leafcheck.pretoria_baseline --output runs/field_v10/pretoria-baseline-replay`
after restoring the expanded intake. The course and supplementary evaluation
images were outside this comparison.

### Matching the corn intake to the study

The [Rabbany et al. release](https://data.mendeley.com/datasets/sb4wwjs8bz/1)
provided 349 verified original JPEGs in the three suitable source folders.
The reviewed sample mainly shows detached leaves with pale backing.
The screening retained 80 healthy, 17 common-rust, and 12 gray-leaf-spot
views. It excluded 240 possible repeats. A seeded visual sample of 18
excluded/keeper pairs showed repeated framing of the same leaves or leaf
sections. This supports the precaution without establishing specimen
identity for every photo.

The [registered corn comparison](../../configs/leafcheck_corn_field_v1.json)
requires at least 50 retained images in every prospective source class.
The intake failed in two classes. The training entry point confirmed the
gate before creating a run or checkpoint. The release is available for
reproduction and remains unused in training or scoring. See the
[screening decision](bangladesh_corn_intake.json).

### Seasonal corn training comparison

The separate [seasonal corn release](https://data.mendeley.com/datasets/vy629dngm8/1)
contains 2,943 originals in its pinned archive. The reviewed sample of
36 shows leaves in place. The common-rust catalog includes 127 filenames
on one day and two without a recorded date. The intake kept the source
together in training to avoid inferring site or specimen independence.
A seed-independent archive-path hash chose 429 original files, including
all 129 common-rust records. The overlap audit retained 412 images.
It excluded 17 possible repeated views. The release contains 126
common-rust, 144 gray-leaf-spot and 142 healthy photos.

Both random-initialization arms used 3,364 master corn training images
and 412 extra occurrences. The control used matched class occurrences
from the master. The treatment used the distinct source originals.
The independent Pretoria development set supplied the four-label
macro-F1 objective. All six corn condition outputs were eligible.
The master guard used all six classes over 959 validation images.

| Method with corn supplied | Pretoria correct | Pretoria macro-F1 | Master correct | Master macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Deployed CNN restricted to corn labels | 26/94 | 0.202778 | 906/959 | 0.913212 |
| Matched scratch control | 26/94 | 0.238126 | 793/959 | 0.706426 |
| Scratch candidate with the seasonal photos | 22/94 | 0.221419 | 791/959 | 0.715208 |

The control stopped after epoch 15 and selected epoch 10. The treatment
stopped after epoch 11 and selected epoch 6. Both met the registered
patience stopping rule and reloaded the selected checkpoints. The
treatment gave −0.016707 macro-F1 against the matched control and
+0.018642 against the frozen CNN given corn. It missed both minimum
gains and the master limits. It correctly named 2/24 common-rust photos
against the control's 0/24, while northern-leaf-blight recall was 7/22
against 14/22. The experiment ended after one seed. The 177-label
deployment and the course and supplementary evaluations remain fixed.

[Selection, per-class results and training curves](corn_seasonal_study.json).
Restore the [source and training instructions](../../data/README.md#seasonal-corn-training-study)
to rerun the matched comparison.

### High-score corn upload replay

The deployed threshold suggested labels for 42 of the 94 Pretoria
development photos. Nine suggestions were jointly correct. At a
descriptive 0.90 score cutoff, three of eight photos were correct.
All three correct photos at that cutoff carried the northern-leaf-blight
reference label. These are development counts. They do not support
changing the application threshold.

The nine highest-score errors included close-ups with visible leaf
markings, occlusion, and two healthy proxies labeled as northern leaf
blight. The packaged app's upload endpoint replayed three original
bytes. It scored common rust as gray leaf spot at 0.954, scored a
healthy proxy as northern leaf blight at 0.916, and correctly scored
one northern-leaf-blight photo at 0.970. All 177 scores matched standalone
inference exactly. The invalid-byte request returned HTTP 400.
[Upload evidence](pretoria_upload_replay.json)
