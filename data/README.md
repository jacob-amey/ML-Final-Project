# LeafCheck dataset splits

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
