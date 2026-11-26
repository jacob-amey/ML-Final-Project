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
