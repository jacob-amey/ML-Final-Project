# LeafCheck datasets

## Final model: 177 labels

Use `data/processed/leafcheck177_final/` for the final custom CNN. This release
preserves all 177 master labels and retains 84,269 of the 90,574 master images.
The original master folder is unchanged.

| Folder | Images | Purpose |
| --- | ---: | --- |
| `train` | 59,008 | Fit model weights; estimate preprocessing statistics |
| `val` | 16,836 | Select settings and checkpoints |
| `test` | 8,425 | Final evaluation after development decisions are fixed |

The release removes exact byte/pixel duplicates, confirmed cross-label related
groups, images with a side below 32 pixels, and recorded unresolved/content
issues. It retains traceable images without documented specimen IDs and assigns
SHA-256 image IDs. Available documented leaf IDs, filename families, previous
review evidence, and conservative near-copy matches determine related groups.
Groups stay within one split. Previously used experiment images and recognized
offline augmentation families stay in training.

All three splits contain every label, but some classes have only one test image.
Missing specimen metadata and unreviewed low-priority similarity candidates mean
that additional related views or annotation errors may remain. Consult the
[release README](processed/leafcheck177_final/README.md),
[per-class counts](processed/leafcheck177_final/split_summary.csv), and
[recorded limitations and checks](processed/leafcheck177_final/dataset_info.json).
Use its `class_to_idx.json`; the final label order has 177 entries.

The [preparation script](../scripts/prepare_final_dataset.py) provides `audit`,
`group`, `prepare`, `verify`, and `restore` phases. The release includes its evidence
snapshots, review decisions, complete manifest, and exclusions. The processed
images remain outside version control under the existing repository rules.

### Restore the exact final split

The submission archive contains release metadata under
`source/data/release_metadata/leafcheck177_final/`. From its extracted
`source/` directory, with the research environment installed and the original
master directory available, run:

```sh
.venv/bin/python scripts/prepare_final_dataset.py restore \
  --master /absolute/path/to/Plant_Dataset_Updated_2026-11-11 --workers 8
.venv/bin/python scripts/prepare_final_dataset.py verify --workers 8
```

Restoration uses the fixed manifest to copy files from the original master into
the exact saved assignments. It checks the archived metadata hashes, master
manifest, copied bytes and split groups before publishing the restored release.
For a copy of the already prepared release, place it in
`data/processed/leafcheck177_final/` and run only `verify`. The portable demo
uses its own packaged weights and two evaluation images.

The local restore verified all 84,269 images. A recipient still needs permission
and access to the exact original master or frozen release bytes to retrain.
The upstream links below identify sources, and archive metadata preserves the
specific membership and hashes from the merge. A fresh upstream download alone
does not reproduce that merge. With the submission archive alone, run
`python3.14 scripts/verify_saved_evaluation.py` from `source/` to reproduce
the saved metrics using the archived predictions and metadata.

## Tomato and potato blight supplement

`data/processed/leafcheck_blight_v1/` contains additional online photos for a
follow-on training and validation study. It uses the same class IDs for tomato
late blight, tomato early blight, healthy tomato, and potato late blight.
This later collection did not contribute training or evaluation images to
the frozen CSC-427 result.
See the [release notes](../docs/blight_dataset_v1.md) for sources, split groups,
overlap checks and rebuild commands.

## Field expansion and source-held validation

The [field expansion study](../docs/field_expansion_study.md) uses additional
publisher-original training photos and reserves a separate Taiwanese tomato
collection for testing. The [source registry](manifests/leafcheck_field_v2_sources.json)
pins reuse terms, exact archive hashes and label mappings. This is a separate
follow-up release with the original 177-class mapping.

## Independent potato evaluation and WASCAL training

The [crop-context study](../docs/potato_context_study.md) uses two later
releases. `leafcheck_okara_v1` contains 3,407 potato images for reserved
evaluation. Its [release seal](manifests/leafcheck_okara_v1_release.json) and
[source registry](manifests/leafcheck_okara_v1_source.json) identify the
screening, labels and public mirror bytes. The exact images are separate from
all training releases.

`leafcheck_tom2024_v1` contains 578 training images from the TOM2024
collection under CC BY 4.0. Its three compatible labels remain in the
existing class map. Check the [source record](manifests/leafcheck_tom2024_v1_source.json),
[per-file checksums](manifests/leafcheck_tom2024_v1_files.csv) and
[release seal](manifests/leafcheck_tom2024_v1_release.json) before use.
Restore the prior field releases and the frozen master first. Build Okara
before TOM2024 so its protected membership is present for overlap screening.
The build commands expect absent output directories.

```sh
.venv/bin/python scripts/prepare_okara_holdout.py fetch
.venv/bin/python scripts/prepare_okara_holdout.py build
.venv/bin/python scripts/prepare_okara_holdout.py verify
.venv/bin/python scripts/prepare_tom2024.py fetch
.venv/bin/python scripts/prepare_tom2024.py build
```

The [Bangladesh tomato collection](https://data.mendeley.com/datasets/jttrv2w27r/1)
contributes 508 test-only images across five labels, including both blights.
It is separately licensed CC BY 4.0. Restore the prior field releases and
TOM2024 first. The [registry](manifests/leafcheck_bangladesh_tomato_v1_source.json)
and [file checksums](manifests/leafcheck_bangladesh_tomato_v1_files.csv) pin the
publisher inputs. Match the [release seal](manifests/leafcheck_bangladesh_tomato_v1_release.json)
before evaluating.

```sh
.venv/bin/python scripts/prepare_bangladesh_tomato_holdout.py fetch
.venv/bin/python scripts/prepare_bangladesh_tomato_holdout.py build
```

## PlantSeg annotation experiment

All 7,774 publisher images matched originals already in the frozen master.
The match used names, publisher ZIP member sizes and CRC32 values, and
recorded SHA-256 values from the master. The original PlantSeg test list
includes 1,081 files assigned to LeafCheck training. Treat the published
PlantSeg split as an already exposed benchmark for this model. The
[source audit](../docs/potato_context/plantseg_source_audit.json) lists the
observed roles. [Publisher record](https://zenodo.org/records/17719108)

The [derived region release](manifests/leafcheck_plantseg_regions_v1_release.json)
uses original images and annotations present in both designated training
splits with compatible labels. It records parent IDs, crop geometry, and
mask checksums. These views reuse existing parents and do not add independent
originals. The metadata includes the source URL and CC BY-NC or CC0 tag.
Keep the media under their applicable source terms.

```sh
.venv/bin/python scripts/prepare_plantseg_crops.py fetch
.venv/bin/python scripts/prepare_plantseg_crops.py audit
.venv/bin/python scripts/prepare_plantseg_crops.py build
```

The commands use checksum-pinned byte ranges of the publisher archive.
Existing restored master files are required. Supply `--master /absolute/path`
to the audit and build commands when the recorded master location differs.
The archive is sparse and contains the annotations and directory records.
It cannot restore the original master photos.

## Camera dated Pakistan tomato development group

The [publisher archive](https://data.mendeley.com/datasets/3mbnb82mxd/2)
has a CC BY 4.0 license and includes raw and augmented folders. Only the
158 raw files bearing the November 2, 2025 camera date were considered.
An inspected sample showed leaves on plants under natural conditions.
The [intake registry](manifests/leafcheck_pakistan_tomato_intake.json)
pins the publisher archive, partial byte ranges and label mapping. The
[release seal](manifests/leafcheck_pakistan_tomato_dev_v1_release.json)
records 142 admitted photos after date, overlap and label-conflict checks.
The labels come from the publisher's visual annotations. Specimen
independence within the session remains unknown.

All included files stay together in development. Restore the protected
earlier releases before building this one. Use an absent output directory
for a separate build.

```sh
.venv/bin/python scripts/inspect_pakistan_tomato.py --all-dated
.venv/bin/python scripts/build_pakistan_tomato_development.py
.venv/bin/python -m leafcheck.field_baseline \
  --release data/processed/leafcheck_pakistan_tomato_dev_v1 \
  --output runs/pakistan_tomato_baseline
```

## PlantCity independent confirmation

The [PlantCity v4 source](https://data.mendeley.com/datasets/w8kh2xkspx/4)
provides original Pakistan photos under CC BY 4.0. The release samples
312 originals in 13 compatible folders of the published original archive.
It excludes 13 possible prior matches or related views. The [seal](manifests/leafcheck_plantcity_holdout_v1_release.json)
records 299 eligible test-only images spanning six crops. Specimen IDs
are unavailable. The processed training archive supplied no examples.
Restore using `scripts/prepare_plantcity_holdout.py fetch` and then
`build`. The completed frozen evaluation is recorded under
`docs/field_optimization/`. Exclude the release from development.

## CCMT crop-only training sample

The [CCMT v1 source](https://data.mendeley.com/datasets/bwh3zbpkpv/1)
provides Ghana farm images under CC BY 4.0. The intake samples 32
publisher-listed JPEGs in each of 17 original folders for tomato,
maize and cassava. Maize maps to `corn`. This is a bounded sample
from the public listings. The publisher archive includes a separate
augmented collection. The crop study uses crop labels alone because
conditions such as leaf blight do not map cleanly to LeafCheck labels.
Individual capture groups are unknown. All eligible photos remain
in one training-only source group.

The [pinned registry](manifests/leafcheck_ccmt_crop_v1_files.csv)
provides 544 file IDs and publisher SHA-256 values. The
[release](manifests/leafcheck_ccmt_crop_v1_release.json) admits 529
after prior overlap, repeat-view and image quality checks. Restore the
earlier registered releases first. The registry establishes exact
restoration without repeating the live catalog sample.

```sh
.venv/bin/python scripts/prepare_ccmt_crop_training.py fetch
.venv/bin/python scripts/prepare_ccmt_crop_training.py build
```

## Holeta and TLID independent crop confirmation

[Holeta potato](https://data.mendeley.com/datasets/v4w72bsts5/1) and
[TLID tomato](https://data.mendeley.com/datasets/kt64b2kh89/1) provide
CC BY 4.0 photos. Holeta captures came from an Ethiopian potato farm.
TLID contains greenhouse-grown original leaf views in Brazil. Its
derivative patches are excluded. Only healthy and late-blight potato,
and healthy and bacterial-spot tomato, map into this study. The archive
lists 363 eligible-extension healthy and 67 late-blight members,
while the Holeta description lists 363 and 63. The intake pins the
actual file hash and selected paths.

The [release receipt](manifests/leafcheck_crop_confirmation_v1_release.json)
covers 245 screened test-only files, sampled from 256 candidates.
Each publisher remains one source group. The exact manifest and seal
were reproduced from the pinned files. The set was opened only after
the crop candidate freeze and now stays outside development.

```sh
.venv/bin/python scripts/prepare_crop_confirmation.py fetch
.venv/bin/python scripts/prepare_crop_confirmation.py build
```

## Image Safari site-held crop development

The [Image Safari collection](https://registry.opendata.aws/artemis-image-safari/)
provides original field photos under CC BY-SA 4.0. This sampling uses
paired publisher metadata requiring an original image and agreeing
crop type. The [intake](manifests/leafcheck_imagesafari_intake.json)
assigns whole sites to either training or validation across all seven
crops, then takes a bounded sample from the earliest available dates.
The [pinned release](manifests/leafcheck_imagesafari_crop_v1_release.json)
contains 129 training and 146 validation photos. It excluded one
repeated view after checks against protected prior images.

These photos carry crop labels only. Some show fields, sparse plants,
vegetation with multiple leaves, or soil rather than diagnostic leaf
detail. Keep this cohort's crop results separate from crop-and-condition
accuracy. The metadata does not establish individual specimen identity.
The source registry identifies both image and sidecar objects and their
SHA-256 values. Raw images remain local.

```sh
.venv/bin/python scripts/prepare_imagesafari_crop.py restore
.venv/bin/python scripts/prepare_imagesafari_crop.py build
.venv/bin/python -m leafcheck.imagesafari_crop_baseline \
  --selection runs/field_v6/crop-434-recovered/decision.json \
  --output runs/field_v7/imagesafari-baseline
```

## Hossain tomato context confirmation

[Hossain et al.](https://data.mendeley.com/datasets/93h9p62kg4/1) publish
tomato photos with visual condition labels under CC BY 4.0. We acquired
all 473 JPEGs from the three compatible publisher folders. Their EXIF
capture day is February 25, 2025. The visual sample showed detached leaves
on pale surfaces. The confirmation therefore represents a single
acquisition group under controlled framing.

The [pinned release](manifests/leafcheck_hossain_tomato_v1_release.json)
retained 181 early-blight, 98 healthy, and 106 late-blight photos. Checks
excluded 72 likely repeat views, seven possible prior matches and nine
entries from conflicting-label clusters. The cohort stayed outside all
fitting and selection. Its opened predictions can be recomputed from
the [saved records](../docs/field_optimization/tomato_context_predictions.csv).

```sh
.venv/bin/python scripts/prepare_hossain_tomato_holdout.py fetch
.venv/bin/python scripts/prepare_hossain_tomato_holdout.py build
```

## Earlier three-class dataset

The dataset assigns `early_blight`, `healthy`, and `late_blight` to class indices 0, 1, and 2 respectively. The image folders under `data/processed/leafcheck/` are ready for a loader such as PyTorch `ImageFolder`. The [image inventory](manifests/image_inventory.csv) records the exact membership, original path, source revision, group, file hash, and any exclusion. The [label mapping](manifests/label_mapping.csv) records how the source labels translate to project labels.

| Folder | Source | Early blight | Healthy | Late blight | Total | Use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `train` | PlantVillage | 628 | 619 | 600 | 1,847 | Fit model weights and apply training augmentation |
| `validation` | PlantVillage | 156 | 156 | 148 | 460 | Select checkpoints and tune settings |
| `test` | PlantVillage | 215 | 224 | 172 | 611 | Measure final performance on the development source |
| `external_test` | Cropped PlantDoc | 75 | 62 | 103 | 240 | Measure final performance on a different image source |

The internal splits contain 462 training, 115 validation, and 153 test leaf groups. Use the training and validation sets while developing the model. Evaluate both test sets after locking the model and any decision threshold. Report the two test results separately.

## PlantVillage selection and split

All 4,500 official color files for these classes matched the local download byte for byte against the upstream Git blob hashes. The 26 other files in the local tomato folders matched renamed PlantDoc files and are marked as excluded local copies. The manifest uses the PlantVillage Hugging Face revision for split and leaf metadata and the upstream GitHub tree for the original color bytes. [PlantVillage split and leaf metadata](https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7), [PlantVillage color files](https://github.com/spMohanty/PlantVillage-Dataset/tree/7f7ecc7e1eaca78107e3affe7cb5abd9427e139a/raw/color)

Only images with an unambiguous leaf ID in the release metadata were eligible for internal splits. This excluded 1,582 images. Five byte-identical pairs among those excluded images span the published train and test lists. All eligible images on the published color test list stayed in `test`. For each class in sorted project-label order, the preparation process sorted the eligible published training leaf IDs, shuffled them with one Python `random.Random(434)` instance, and assigned the first `round(0.20 × number of leaf IDs)` groups to `validation`. The rest went to `train`. This produced an internal image split of approximately 63.3% training, 15.8% validation, and 20.9% testing. Grouping is at the documented leaf level. Plant and capture-session relationships may still be missing.

## External test selection

The three compatible classes provided 262 candidates across the Cropped PlantDoc release's original train and test directories. Both directories were eligible solely for external evaluation in this project. All candidates matched original filenames, class labels, and release split assignments in the companion detection annotations. The generic `Tomato leaf` label serves as the healthy proxy. It reflects the source annotation and carries no independent diagnosis. Many source files are wider plant views, so this benchmark also tests changes in framing, lighting, and background. [Cropped PlantDoc release](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8), [Companion annotations](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset/tree/4730a233a555b30ee98e0879c63ad25d82407455)

The 22 exclusions covered 10 repeated or closely related views, four comparison graphics or charts, three images with visible disease-name text, two images with ambiguous source titles, two copies of one photograph with conflicting labels, and one illustration. For same-label repeated views, the largest image by pixel area was retained, with source path breaking ties. The exclusion reason is recorded for every omitted image. The review used byte hashes, decoded-pixel hashes, low-resolution visual hashes, and visual inspection. The checks can miss heavily cropped copies, additional related views, and label errors.

## Reusing the splits

Use only manifest rows with `status=included`. `image_path` is relative to the repository root, `source_path` is relative to the named release or local bundle, and `source_split` is the publisher's split where available. `split` is the assignment for this project. External working filenames use the first 16 characters of the SHA-256 digest for portability. The copied bytes are unchanged.

The repository ignores `data/processed/`. To restore it in another checkout, retrieve each included `source_path` from the pinned source, verify its SHA-256 against the inventory, and place it at the corresponding `image_path`. The verified local PlantVillage copy can also supply those images. Its folders are named `Tomato__early_blight`, `Tomato__healthy`, and `Tomato__late_blight`. Match both the filename and hash. Review each source's attribution and reuse terms before redistributing image files. [PlantVillage dataset card](https://huggingface.co/datasets/mohanty/PlantVillage), [PlantDoc repository license](https://github.com/pratikkayal/PlantDoc-Dataset/blob/5467f6012d78d1c446145d5f582da6096f852ae8/LICENSE.txt)

### Unsupported upload development photos

The [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html)
sampling registry `manifests/leafcheck_unsupported_openimages_v1_files.csv`
pins the image IDs, asset links and per-image attribution records. The
screened manifest `manifests/leafcheck_unsupported_openimages_v1_manifest.csv`
preserves the 45 included photos, three exclusions and exact hashes.
Restore with `.venv/bin/python scripts/prepare_unsupported_openimages.py fetch`.
The restore checks the mirror bytes against the manifest. The project keeps
the photo bytes local. These photos are a development challenge for
unsupported-input checks.

```bash
.venv/bin/python -m leafcheck.unsupported_screen --energy --output runs/unsupported-replay
.venv/bin/python -m leafcheck.upload_support_probe --output runs/support-probe-replay
```

### Field intake and partial-label potato training

Read the [intake and training record](../docs/field_optimization/field_v9_intake.md)
for provenance and the matched comparison. The 305 IIT Mandi images were
used only in the training arm. The [source files](manifests/leafcheck_iit_potato_intake_files.csv)
and [audit](manifests/leafcheck_iit_potato_intake_audit.csv) record the exact
bytes and the original coarse labels. The master and registered study splits
stay fixed.

```bash
.venv/bin/python scripts/audit_field_preflight.py fetch --iit-potato
.venv/bin/python scripts/audit_field_preflight.py audit --iit-potato
.venv/bin/python -m leafcheck.potato_partial_study \
  --audit runs/field_v9/iit-potato --output runs/field_v9/potato-partial-replay
```

### Seasonal corn training study

The [seasonal corn archive](https://data.mendeley.com/datasets/vy629dngm8/1)
uses CC BY 4.0. The [release seal](manifests/leafcheck_seasonal_corn_v1_release.json)
and [manifest](manifests/leafcheck_seasonal_corn_v1_manifest.csv) record 412
admitted photos and 17 possible repeat views excluded. The original
source is training-only for this comparison. Pretoria is a separate
frozen development cohort and was checked for overlap before admitting
training photos.

```bash
.venv/bin/python scripts/audit_pretoria_intake.py --expanded --output runs/pretoria-restore
.venv/bin/python scripts/prepare_seasonal_corn.py
.venv/bin/python -m leafcheck.corn_field_study \
  --recipe configs/leafcheck_corn_seasonal_v1.json \
  --training-addition data/processed/leafcheck_seasonal_corn_v1 \
  --output runs/corn-seasonal-replay
```

### Reserved Colombian coffee evaluation source

The [coffee intake audit](../docs/field_optimization/coffee_holdout.md) explains
the archive reconciliation, exact source IDs, two condition labels and overlap
screen. Restore the established releases and the Pretoria originals, then use
`.venv/bin/python scripts/prepare_decafia_holdout.py`. Keep the resulting
`leafcheck_decafia_v2` release outside fitting and routine validation.
