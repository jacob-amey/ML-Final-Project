# Final 177-label dataset preparation

Prepared from the local `Plant_Dataset_Updated_2026-11-11` master snapshot on
2026-11-15. This document records dataset preparation and loading checks; it
contains no final CNN training or evaluation results.

## Release and counts

The prepared release is `data/processed/leafcheck177_final/`. The original
90,574-image, 177-label master remains available as the source snapshot. The
release retains 84,269 images (93.04%) and all 177 labels.

| Split | Images | Purpose |
| --- | ---: | --- |
| Training | 59,008 | Learn weights and estimate preprocessing statistics |
| Validation | 16,836 | Select settings, checkpoints and decision thresholds |
| Test | 8,425 | Final reporting after development decisions are fixed |

Every split contains all 177 labels. Class indices are alphabetical and defined
by the release's `class_to_idx.json`. Files retain their original bytes and
resolution. The audit fully decoded the first frame, matching the intended
image loader, rather than checking image headers alone.

## Exclusions

| Reason | Images |
| --- | ---: |
| Repeated byte-identical files | 1,277 |
| Additional exact decoded copies under rotation/reflection | 4,275 |
| Related groups with conflicting class annotations | 711 |
| Width or height below 32 pixels | 38 |
| Unresolved cross-label visual-review case | 2 |
| Previously documented content or annotation issues | 2 |
| **Total** | **6,305** |

The complete manifest accounts for every master image, including exclusions,
original labels, source paths, checksums and final assignments. Excluded source
images were omitted from the prepared copy, not deleted from the master.

## IDs, grouping and leakage precautions

The agreed policy preserves traceable images without documented plant/leaf IDs.
Each retained image receives a stable SHA-256 content ID. A content ID identifies
an image file; it cannot establish which physical plant or leaf was photographed.
There are 31,244 retained images with matched documented PlantVillage leaf IDs
and 53,025 with traceable source records but no documented leaf ID.

The 48,449 retained groups combine master duplicate/augmentation metadata,
documented leaf IDs, conservative filename leaf families, the earlier project
review inventory, exact decoded pixel identity, and strict or visually reviewed
near-copy evidence. Known groups stay within a single split. Groups containing
recorded earlier-experiment images or recognized offline flip augmentations are
training-only. This keeps the new holdouts separate from those recorded model
experiments. Additional unrecorded work cannot be inferred from the available
files.

The seed is 434. Allocation aims for 70%/20%/10% within each class, prioritizing
whole groups, training representation and fresh holdouts over exact percentages.
The earlier stricter policy requiring documented leaf IDs is not applied here.

## Visual similarity review

The rotation/reflection-aware perceptual screen found 66,928 candidate image
pairs. A perceptual hash describes broad visual appearance; a matching hash does
not establish image identity. Candidate pairs also underwent aligned 64-pixel
thumbnail comparisons. Same-label candidates were automatically grouped only
at mean absolute error at most 1 and root mean squared error at most 3 on the
0–255 intensity scale. These nonexact views remain available within one split.

Visual review covered 310 representative pairs: 179 were confirmed as the same
source photograph, 130 were different images, and one pair remained unresolved.
Confirmed same-source cross-label groups were excluded without guessing the
correct diagnosis. The unresolved pair's existing groups were also excluded.

Review priorities covered all strict cross-label matches, additional component
pairs with mean error at most 8 or identical perceptual hashes, and remaining
component pairs outside the common rice/dark-background-leaf families. The
release retains a list of 792 lower-priority, unreviewed cross-label component
pairs. These are largely narrow rice leaves or leaves on shared backgrounds.
Similarity alone did not justify discarding them. Crops, edits, unmarked related
views and annotation errors may remain.

## Verification performed

- Fully decoded and checksum-verified all 90,574 master image files. All decoded
  successfully; 38 failed the minimum-size policy.
- Copied and checksum-verified every retained image, then rechecked all copied
  file hashes before publishing the release.
- Checked unique retained byte hashes and decoded rotation/reflection identities.
- Reconciled all 90,574 records: 84,269 included and 6,305 excluded.
- Independently checked 260,532 grouping/content evidence keys, including master
  groups, documented leaf IDs, filename families and prior reviewed groups;
  none crossed splits.
- Checked that recorded earlier-experiment image hashes do not occur in the
  validation or test sets.
- Loaded every split using PyTorch `ImageFolder` and verified the class mapping.
  Sampled a shuffled eight-image batch per split with orientation correction,
  frame-zero selection, RGB conversion, resizing to 128×128 and scaling to [0,1].
  All produced finite tensors shaped `[8, 3, 128, 128]` and integer class labels
  in 0–176. This resolution was a loading check, not a selected final-model setting.

## Limits and next model decisions

There are 69 classes with fewer than ten test images and 83 with fewer than ten
test groups. `broccoli__ring_spot` and `coffee__black_rot` each have only six
images, allocated 4/1/1. One correct prediction therefore gives 100% accuracy for
either class on its single test image; it supplies little evidence of reliable
generalization. Report per-class counts alongside precision, recall and F1.
Additional traceable, independently collected examples should target these
rare classes before assuming that all labels can be learned dependably.

Labels inherit the master source-folder annotations. Generic conditions and
potentially overlapping categories remain separate; this audit does not verify
biological diagnoses. Source-specific backgrounds, web graphics and other
shortcuts may still occur. This release also does not establish reuse rights for
every upstream image. Preserve and cite source information when preparing the
report and sharing data.

Use training data alone for learned normalization statistics, apply augmentation
only during training, and keep the new test set out of model selection. The
final CNN must use the 177-entry mapping and output 177 logits. Training from
random initialization remains the assignment's main model path.

## Reproducibility and evidence

The preparation script is `scripts/prepare_final_dataset.py`, with `audit`,
`group`, `prepare` and `verify` phases. The release includes the full manifest,
exclusions, per-class split/group counts, review decisions, exposure snapshot,
original-to-standardized label mapping, and pinned upstream metadata. Its
preparation snapshot fingerprints the policy and local review evidence.

The master manifest SHA-256 is
`9cdffdb06ce9222d58c0ac1fdcaabe1988c1e0ebebd4c7086f85a308c1925ba5`.

Relevant pinned primary sources:

- [PlantVillage leaf metadata](https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7).
- [PlantDoc source tree](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8).

Dataset preparation and verification were assisted by Codex. The student should
review these policies, understand the grouping and split decisions, and describe
this assistance accurately in project documentation under course rules.

## Supplemental integrity review during final training

After the release freeze, all366 cross-split representative pairs from the792
remaining lower-priority candidates were visually inspected. The first36 were
prioritized by aligned mean absolute error; the additional330 were reviewed as
paired, orientation-aligned contact sheets without disease-label display or model
outputs. An additional seven rice pairs received larger source-image inspection and geometric
feature checks, with one visually distinct pair and a known same-photo software
control. Sparse/background or repeated-point matches did not confirm a new shared
photograph. Two earlier larger views are retained in the initial36-case record.
The software control recovered113 geometric inliers.

No additional photographic identity was confirmed, so the frozen release remains
84,269 images and177 labels with the same fingerprint and split assignments.
The original 792-row candidate file remains intact. The subsequent within-split
review below completes its remaining 426 representatives. Unknown physical
specimens/sessions, unlisted related views, missed crops/edits and inherited
annotation semantics remain limitations. A failed feature match is not proof
of biological independence.

`docs/additional_integrity_review.csv` and its JSON summary record the cases,
methods, source dimensions, control and limitations. The inspection environment
was isolated from model training. Only raw-file integrity was reviewed; test
predictions, metrics and mistakes were not used for model development.

To reproduce this supplemental inspection after restoring the exact release:

```sh
.venv/bin/python -m venv runs/final177/integrity_env
runs/final177/integrity_env/bin/python -m pip install -r docs/integrity_requirements.txt
runs/final177/integrity_env/bin/python scripts/review_integrity_candidates.py --start 0 --stop 366
runs/final177/integrity_env/bin/python scripts/review_integrity_candidates.py --native-cases 673 757 284 286 351 477 525 556 482
runs/final177/integrity_env/bin/python scripts/check_integrity_features.py
```

The helpers reconstruct the original ordered792-case list from release metadata;
they do not require an untracked setup script. The first command creates a
separate inspection environment. Its pinned packages do not replace training
dependencies. Candidate feature matches are review aids, not automatic exclusion
or diagnosis rules. The same-photo control is software evidence only.

### Completion of within-split candidate review, November 17

All remaining 426 within-split representatives were inspected on 36 contact
pages with disease labels and predictions masked: 374 train/train, 40 val/val
and 12 test/test pairs. Fifteen closer cases received larger raw-image views
with original dimensions recorded, plus SIFT/RANSAC correspondence checks in
the isolated inspection environment. The 444 distinct source files matched their
recorded raw hashes. No additional shared photograph was confirmed.

Some high nominal correspondence counts were degenerate: case 444's 34 inliers
and case 331's 20 inliers each collapsed to one destination. Larger source views
and the displayed matches did not establish photographic identity. The same-photo
crop/resize/JPEG software control again recovered 113 inliers at 104 distinct
destinations; it is not a training example or CNN evaluation result.

No images, labels, groups, split membership or release fingerprint changed.
All 792 representatives in this lower-priority list now have supplemental
contact review (366 cross-split and 426 within-split). This is a bounded
identity screen, not exhaustive biological independence or expert label validation.
Raw test files were inspected only for permitted integrity review; test-model
predictions, scores and errors remain excluded from model development.

Evidence: `docs/within_integrity_review.csv` and
`docs/within_integrity_review.json`. Reproduce the raw views and geometry with
the same isolated environment and restored release:

```sh
runs/final177/integrity_env/bin/python scripts/review_within_candidates.py
runs/final177/integrity_env/bin/python scripts/review_within_candidates.py --cases 327 226 669 710 648 444 342 344 671 331 381 780 185 352 129
runs/final177/integrity_env/bin/python scripts/check_within_integrity_features.py
```

These helpers write into a separate `within` evidence folder and preserve the
earlier cross-split review. They never edit the dataset or select a model.
