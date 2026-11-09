# Working proposal outline

This is an outline, not the final two-page submission.

**Working title:** LeafCheck: Evaluating Tomato Leaf Classification Across Image Sources

**Team members:** TBD  
**Proposal deadline:** TBD

## Introduction

Describe the motivation for classifying leaf conditions and the challenge of
moving from controlled images to more varied photographs. Support background
claims with cited research. Scope the tool as a limited research prototype.

## Methods

Planned comparison: a small CNN trained from scratch, pretrained MobileNetV2 with
basic augmentation, and MobileNetV2 with stronger augmentation. Keep the latter
two experiments comparable except for augmentation. Framework choice is pending.

Research question: Does transfer learning and stronger training augmentation
improve classification of selected tomato leaf conditions on another image source?

## Dataset

PlantVillage color images for development and internal testing; compatible
cropped PlantDoc images for external evaluation. Final classes, usable counts,
versions, and exclusions are pending the audit in `data/README.md`.

## Goal / completion criteria

- Finalize a documented shared class set and reproducible grouped splits.
- Train and compare the planned methods.
- Evaluate internal and external test sets using macro-F1, accuracy, per-class
  results, and confusion matrices.
- Deliver a local image-upload demo with side-by-side predictions.
- Submit the documented repository, report, and presentation with contributions.

Do not promise a particular accuracy or assume augmentation will improve results.

## Plan

| Milestone | Owner | Target date |
| --- | --- | --- |
| Data audit and proposal | TBD | TBD |
| Baseline and transfer-learning pipeline | TBD | TBD |
| Controlled experiments and evaluation | TBD | TBD |
| GUI, report, and presentation | TBD | TBD |

## References to review

- Mohanty et al., “Using Deep Learning for Image-Based Plant Disease Detection,”
  2016. https://doi.org/10.3389/fpls.2016.01419
- Singh et al., “PlantDoc: A Dataset for Visual Plant Disease Detection,”
  2020. https://doi.org/10.1145/3371158.3371196

Verify bibliographic details and format final citations in IEEE style.
