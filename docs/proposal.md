# LeafCheck current project scope

Updated November 15, 2026 to reflect the authorized final scope. This supersedes
the older three-class transfer-learning outline. Instructor approval and the
team's formal two-page proposal submission have not been verified.

## Introduction

LeafCheck studies a compact custom CNN trained from random initialization for
177 plant-condition labels. An uploaded photograph is the input; one supported
label and softmax scores are the output. This is a research classification
prototype, not a definitive diagnosis. Labels include diseases, pests, healthy
tissue and generic conditions on leaves, fruit and stems.

PlantVillage illustrates the potential of CNNs on controlled plant images and
also reports a substantial accuracy drop on differently collected images [1].
PlantDoc supplies more varied photographs [2]. These motivate grouped splitting,
source-aware limitations and conservative deployment claims.

## Methods

Implement a small convolutional reference and a deeper custom residual CNN with
standard PyTorch layers. Both start with newly initialized weights; the main
solution uses no pretrained weights. Compare them on fixed splits with other
training settings held constant. Investigate imbalance handling when validation
evidence supports it. Select by validation macro-F1, with lower validation loss
as a tie-breaker. Keep test predictions out of every development decision.

The simple rule always predicts the most frequent training class. Score it and
the frozen selected model on the same test images. Preserve the earlier pilots
and Create ML model in the local comparison website with explicit coverage and
exposure caveats; they are not a controlled full-label benchmark.

## Dataset

The audited release contains 84,269 images across all 177 labels: 59,008 training,
16,836 validation and 8,425 test images. It removes confirmed conflicting groups,
exact copies and recorded unusable images, preserves traceable source records,
assigns content IDs and keeps available related-image groups together. Recorded
earlier-experiment images are training-only in this release.

Some classes have only six images and one test example. Many photos lack
documented specimen IDs. Source labels are not independently diagnosed, and
classification cannot localize several plants/diseases within one photo.
See `data/README.md` and `docs/final_dataset_audit.md` for provenance and limits.

## Goal and completion criteria

- Finish selected training on all designated training images; verify resume and
  checkpoint reload.
- Record controlled experiments and freeze model/preprocessing before testing.
- Report correct/total, accuracy, macro-F1, per-class metrics, supports,
  confusion matrices, examples and inference runtime beside the simple rule.
- Deliver a fingerprinted standalone model bundle running in the existing app.
- Rehearse a fresh upload, one success, one mistake and a justified improvement;
  document actual contributions and assistance.

No accuracy target is promised. The contribution is the implemented, audited,
reproducible local system and evaluation, rather than a novel learning algorithm.

## Work division and deliverables

The user identified their responsibilities as dataset, CNN, training, evaluation
and model handoff, and their partner's as frontend/application/interface.
Partner name, meeting attendance and formal approvals have not been supplied.
Final execution is assisted by Codex and must be acknowledged accurately.

The course also requires a 5–10 page report and a presentation. The team must
review the evidence, prepare its submission and understand the design for Q&A.
See `docs/contributions.md` for attributable work and `docs/final_model_requirements.md`
for requirement coverage. Binary storage follows the repository's rules.

## References

[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for image-based
plant disease detection,” *Frontiers in Plant Science*, vol. 7, art. 1419, 2016,
doi: [10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419).

[2] D. Singh *et al.*, “PlantDoc: A dataset for visual plant disease detection,”
in *Proc. CoDS-COMAD*, 2020,
doi: [10.1145/3371158.3371196](https://doi.org/10.1145/3371158.3371196).
