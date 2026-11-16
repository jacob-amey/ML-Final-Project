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
