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
