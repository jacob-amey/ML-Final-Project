# Custom CNN starter dataset

A small, balanced three-class tomato image-classification dataset for your first custom CNN implementation. It contains 300 unique images: 100 per class. Fifteen images would provide a very small implementation check; the main 300-image set provides separate training, validation, and testing subsets for an initial experiment. This is a prototype dataset, not evidence of performance on the full 177-class task.

## Classes and split

| Label ID | Class folder | Training | Validation | Testing | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | tomato__early_blight | 70 | 20 | 10 | 100 |
| 1 | tomato__healthy | 70 | 20 | 10 | 100 |
| 2 | tomato__late_blight | 70 | 20 | 10 | 100 |
| **Total** | | **210** | **60** | **30** | **300** |

All images are RGB JPEGs, 256 × 256 pixels. Folder names are the classification labels. Class IDs are local to this three-class dataset and differ from the full dataset IDs. `manifest.csv` includes both IDs and links each image to its location in the parent dataset.

## Folder layout

```
Plant_CNN_Starter_300/
  train/tomato__early_blight/   (70 images)
  train/tomato__healthy/        (70 images)
  train/tomato__late_blight/    (70 images)
  val/<same three classes>/    (20 per class)
  test/<same three classes>/   (10 per class)
  quick_check_15/<same three classes>/ (5 per class, copied from train)
```

## First implementation

1. Point your image-folder loader at `train`, `val`, and `test` separately. Each has the same three labels.
2. The files are already a uniform 256 × 256 RGB shape. You can resize to 128 × 128 when loading to reduce computation. Scale pixel values to 0–1; keep preprocessing consistent across splits.
3. Configure the CNN for three input channels and three output classes. A batch size of 16 is a reasonable small starting choice.
4. Train on `train`, use `val` to choose the model and training settings, and use `test` for the final evaluation. Do not augment validation or test images. If using augmentation, apply it only to training images after loading.
5. For the smallest implementation check, point the loader at `quick_check_15`. This contains 15 training images only, suitable for checking loading and deliberately overfitting a tiny batch. It has no independent validation or testing subset. Do not combine it with the main folders, because those images are already in `train`.

For an initial experiment, try a few epochs and check whether training loss falls. Ten test images per class give a very rough result, and accuracy changes in steps of about 3.33 percentage points overall. Expand the dataset before drawing conclusions about generalization.

## Selection and verification

Images were sampled deterministically with seed 42 from the supplied dataset at:

`/Users/jacobamey/Desktop/Brockport_Class/ML-Final-Project/Plant_Dataset_Updated_2026-11-11`

Original train/validation/test membership is preserved. Known groups containing conflicting labels were excluded using the full parent manifest. Only one image from any source duplicate/augmentation group was selected. All selected images have distinct file hashes and distinct decoded image fingerprints under rotations and flips. This does not identify every possible near-duplicate.

All 300 selected images were fully decoded successfully and checked for RGB mode and 256 × 256 dimensions. Files were copied byte-for-byte, and checksums were verified after copying. No images were resized, recolored, or relabeled. Labels are inherited from the parent dataset, not visually diagnosed. The parent dataset was not changed.

The quick-check folder contains 15 convenience copies of the main training images; the folder therefore has 315 image files in total, representing 300 unique images.

## Records

- `manifest.csv`: starter paths, labels, parent paths, parent labels, grouping IDs, and checksums.
- `class_to_idx.json` and `class_labels.csv`: three-class label mapping.
- `split_summary.csv`: counts per class and split.
- `dataset_info.json`: sampling settings and verification results.
- `quick_check_15/manifest.csv`: correspondence between the 15 convenience copies and main training images.
