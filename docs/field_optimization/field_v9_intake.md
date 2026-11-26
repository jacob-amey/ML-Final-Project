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
