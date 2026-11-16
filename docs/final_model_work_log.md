# Final model work log

Dates are recorded in America/New_York unless a run field explicitly says UTC.
Work below is performed with Codex assistance and must not be described as
independent student implementation.

## 2026-11-15 — requirements and starting state

- Read the build guide, available earlier walkthrough, notebooks and run records,
  current dataset documentation, preparation script and local comparison app.
- Reverified the 84,269-image release and 177-class mapping, copied checksums,
  unique decoded copies, split groups and recorded earlier-experiment exposure.
- Confirmed the 32 GB M2 Max and working MPS environment.
- Preserved existing parent/nested-repository changes and created a task branch.
- Declared validation macro-F1 selection, tie-breaking and test isolation before
  running model experiments. Reconciled the older proposal with current scope.

## 2026-11-15 — verified pipeline and benchmark

- Shared frame-zero, EXIF, RGB, bilinear letterboxing and standardization across
  training/evaluation/inference. Cached only train/validation images for speed.
- Froze dataset fingerprint `4287764ba029d96abaa0532d7e1d14d2656cadc774f445be7785b03b69d2903c`.
- Training-only RGB means: 0.478542, 0.510694, 0.422771; standard deviations:
  0.209555, 0.191101, 0.230226.
- Tiny residual CNN learned 12 training images: 12/12 correct and loss 0.001197
  after 40 updates. Finite gradients, parameter updates and reload passed.
- Five pipeline tests passed, including exact CPU equality between four
  uninterrupted updates and two updates followed by save/resume/two updates.
- Warm MPS float32 benchmark, 128-pixel images, batch 128: plain 253,257-parameter
  CNN 0.0230 seconds/update; residual 724,713-parameter CNN 0.0610 seconds/update.
  Excludes validation/checkpoint overhead. Batch 128 outperformed batch 256 in
  images/second for both. Full training logs will establish real epoch costs.
- Both initial full-data configurations use the same maximum 48 epochs, minimum
  24, patience 10, optimizer/augmentation/seed/resolution and schedule; architecture
  is the controlled change. A validation plateau reduces learning rate first.

## 2026-11-15 — comparison audit and demo integration

- The Create ML source records identify the original master splits. The final
  test includes 5,832 images from its declared training split and 1,683 from its
  validation split. It cannot be a held-out comparison on this test. No new test
  predictions were made for this audit.
- Nine existing demo tests passed before changes, including a fresh standalone
  copy. Relevant pre-existing migration/bundling/UI work was preserved in nested
  commit `22e454f`; the unrelated hosting-file deletion was left out.
- Added the explicit final-model adapter, validated bundle installer, fourth
  model card, saved model-versus-rule score display and comparison caveats.
  The final model stays unavailable until real completed artifacts are installed.
- Added final integration tests for class order, numerical API parity, JPEG EXIF,
  grayscale/RGBA PNG, unavailable artifacts and standalone packaged examples.
  These final-artifact tests are pending selection and evaluation.

## 2026-11-15 — completed reference and supplementary protocol

- Compact reference finished 48 full-data epochs, 22,128 updates, with selected
  epoch 44: validation accuracy 0.854835, macro-F1 0.536531, loss 0.543217.
  Full validation recomputation after reload matched. Total runtime 1,775.1 s.
  Its late learning-rate reductions produced flattening gains; the stronger
  architecture experiment is the next use of resources.
- Validation-source diagnostics showed different results on controlled leaf-ID
  images and a small PlantDoc-matched stratum. Class compositions differ, so
  this is a generalization warning rather than a causal background claim.
- Audited the existing 240-photo external collection without predictions.
  Removed 35 exact decoded/recorded group overlaps plus 22 additional related
  photos confirmed through 26 visual pair reviews. Frozen supplement: 183 photos
  (50 early blight, 54 healthy-proxy, 79 late blight). This covers only three
  tomato labels and is not a completely unseen-source benchmark.
- Registered the supplementary conditional rule using training counts only:
  always predict tomato late blight. Both main and supplementary sets stay
  sealed from model selection until the final configuration is frozen.
- CPU and MPS resume checks passed. Added carried partial-epoch timing so resume
  does not under-report the first resumed epoch's runtime; optimization is unchanged.
