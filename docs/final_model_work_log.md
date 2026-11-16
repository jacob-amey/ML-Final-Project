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

## 2026-11-15 — source checkout and software safeguards

- Found that the app's parent gitlink had neither `.gitmodules` configuration nor
  a nested remote. A normal parent checkout would omit all website source.
- Preserved pre-existing work in nested `22e454f`, new integration in `a8d2eef`,
  verified a Git history bundle, and retained original Git metadata under the
  parent's `.git/leafcheck-demo-history`. The existing uncommitted hosting-file
  deletion remained intact in that archived metadata/worktree record.
- Consolidated app files into ordinary parent-repository source without changing
  their working contents. Original model binaries remain ignored.
- Seven pipeline tests passed, including pixel-cache tamper rejection and CPU/MPS
  resume. Added source/checksum checks when building or loading image caches.
- Eight legacy/API checks passed against the four-adapter server with the final
  model explicitly unavailable; earlier real models continued predicting.
- Added app revision detection so a launcher does not silently reopen an older
  already-running process after code/model-reference updates.
- Standalone inference on the completed reference's validation images matched
  cached batch inference: maximum logit difference 2.29e-5, 177 aligned classes.
- A 100-update concurrent probe of the weighted experiment took 43.84 seconds
  including startup. No throughput advantage was demonstrated; full jobs remain
  sequential. Its partial checkpoint is queued for complete resumable training.

## 2026-11-16 — completed capacity comparison and rejected refinement

- Residual CNN completed 48 full-data epochs (22,128 updates), selected epoch 47:
  validation accuracy 0.882989, macro-F1 0.595327, loss 0.490400. Full selected-model
  reload matched. Runtime 4,188.1 seconds, including session overhead.
- Compared with the compact reference, observed validation macro-F1 increased by
  0.058796 with the same data, seed, augmentation policy, optimizer and schedule.
  Initialization/augmentation random streams need not be identical across
  architectures; this is an observed single-seed comparison, not a significance claim.
- Ran a matched post-training BatchNorm statistics experiment on all 59,008
  training images with augmentation off and learned parameters held bitwise
  fixed. Validation macro-F1 changed from 0.595327 to 0.595177 (-0.000149).
  Rejected the refinement and preserved the original checkpoint. Additional cost
  591.7 seconds; its logged total includes the parent procedure and is not a
  separate fresh 48-epoch training run.
- The class-weighted residual run resumed its exact saved batch-100 state and
  continues full-data training. Its comparison decision remains pending convergence.
- A second initialization of the leading recipe is planned if practical under
  the measured local runtime. The test and supplementary sets remain unscored.

- Recorded training support for all 177 labels: 4–5,116 images per class, median
  100, with 26 classes below 30 images. Saved a reproducible distribution figure
  and known/inferred group counts; these groups do not prove specimen independence.
- Eight pipeline checks passed, including explicit float32 inference inputs under
  a changed PyTorch default dtype. Added a readable supplementary confusion figure
  generator, to run only after the final model is frozen.

## 2026-11-16 — completion safeguards and review materials

- Nine pipeline checks passed (34.5 seconds); completed procedures now reject
  accidental re-training through the resume command while preserving their files.
- Prepared a focused final-review notebook, with no invented outputs, and a
  report generator that refuses incomplete evaluation artifacts. Execution waits
  for the real frozen bundle. Added a delivery-archive fresh-process rehearsal.
- The deeper model's late validation curve still improved through epoch 47.
  Before any test evaluation, allow one separate longer-budget experiment of the
  leading recipe if its 48-epoch curve still shows an improvement opportunity:
  maximum 64 cumulative epochs, same scheduler and early-stop rule. Parent
  checkpoints are preserved. A temporary-copy check confirmed all model, optimizer,
  scheduler and RNG tensors are bitwise preserved, with only declared procedure
  metadata changed. No additional training has run under this option yet.

- Preregistered the remaining bounded search in `final_selection_plan.json`: finish
  the matched loss comparison, at most 64 cumulative epochs for its leader if
  late improvements justify continuation, then one fresh seed of that recipe.
  This is a practical local compute budget, not a global optimality claim.
- A short synchronized timing probe during active training showed no justified
  pipeline shortcut; MPS does not support foreach gradient clipping here. The
  active pipeline remains unchanged. A host-process snapshot was about 4.57 GB
  resident memory; it is not a peak GPU/unified-memory measurement.
- Inspected confidently incorrect validation examples. Scores near one can be
  wrong; visually similar pest damage and mixed plant parts remain limitations.
  No labels were changed from model disagreement or unauthenticated diagnosis.

- Observed the actual browser upload flow against the staging app: an unreadable
  PNG is rejected clearly, and the other three real classifiers still predict
  while the final package is unavailable. Saved preflight evidence and screenshot.
  Simplified the missing-final-package message; the real final-model browser
  rehearsal remains pending and is not inferred from this preflight.

## 2026-11-16 — completed class-weight comparison

- Weighted residual finished 48 full-data epochs (22,128 updates), selected epoch 48:
  validation macro-F1 0.603053, accuracy 0.874376, loss 0.535239, 4,204.0 seconds.
  Reloaded checkpoint reproduced the full validation results. Its macro gain
  over the unweighted model is 0.007727; accuracy decreases by 0.008612.
- Reconciled completed-checkpoint validation diagnostics against saved predictions
  and now stamp every diagnostic with its checkpoint hash. Among 71 labels with
  fewer than 20 validation images, mean F1 rises 0.344437→0.375945, while zero-recall
  labels increase 13→14. Weighting therefore helps the average without solving all
  rare classes. Source-stratum differences are not a separate field benchmark.
- Started `residual177_weighted_long64_seed434` from the exact epoch 49 resume
  state. Its parent remains intact; maximum 64 cumulative epochs, same optimizer,
  scheduler, augmentation, weighting and early-stop policy.

## 2026-11-16 — duration check and independent seed

- The duration check stopped at 55 cumulative epochs after the configured
  validation plateau. Selected epoch 50: macro-F1 0.603681, accuracy 0.876277,
  loss 0.529681. Reload verification passed. Additional training cost: 605.4 s;
  cumulative parent plus continuation: 4,809.4 s. The 0.000628 macro gain is small
  and below the preregistered 0.005 practical signal; no further duration search.
- Declared a fresh seed-435 run of the same weighted recipe and maximum
  64-epoch/early-stop procedure. This is a new initialization and random training
  stream, not a continuation or pretrained model. Both held-out sets remain sealed.

- Installed only the app's two declared top-level dependencies in a separate
  environment from the official PyPI registry (11 installed runtime distributions
  plus pip). Verified the three historical real models and the completed
  177-output development checkpoint can predict with NumPy, torchvision, pandas,
  matplotlib and notebook libraries absent. Top-five development scores matched
  the research runtime exactly on a software fixture; final all-score API parity
  and packaged-archive checks remain pending the frozen selected artifact.

## 2026-11-16 — user timing clarification and resolution check

- The user confirmed there is no hard deadline and asked to prioritize the
  strongest defensible result. Amended the still-sealed selection plan before
  test evaluation to allow one modest resolution check (160 versus 128 pixels).
- The hypothesis is preservation of small lesion detail, not a claim that higher
  resolution will solve source/label or rare-class limitations. Keep architecture,
  data, weighting and training policy fixed; benchmark actual local cost first.
  A stability repeat is conditional on meaningful validation benefit. No open-ended
  architecture/resolution grid or test-driven tuning is introduced.

- Confirmed seed 435 completed64 full epochs, selected63: validation macro-F1
  0.598316, accuracy 0.874258, loss 0.534610; runtime 5,214.1 seconds. Reload verified.
  Its status records completion at 2026-11-16T07:47:14.470996+00:00; completed status was observed
  at 2026-11-16T12:52:04.203999+00:00. The two 128-pixel seeds span 0.005365 macro-F1.
- Declared the resolution acceptance rule before 160 validation: a seed 434 gain
  of at least 0.005365 triggers a160-pixel seed 435 repeat; select 160 only
  if the two-seed mean gain reaches 0.005, otherwise retain 128 for lower compute.
  Checkpoint selection within the accepted recipe keeps the original F1/loss rule.

- Verified the two completed 128-pixel seeds share identical numeric settings,
  ordered labels, preprocessing, per-class loss weights and core training source hashes.
  Overall macro-F1 spans 0.005365; 12 labels change F1 by at least 0.2, and 11 have
  zero observed validation recall in both runs. Saved all 177 class comparisons.
- Registered the full-data Compact177 reference (epoch 44, 253,257 parameters) for
  a fair same-holdout comparison. Its checkpoint will freeze with the selected
  model; test scores cannot change the deliverable. It is separate from the
  earlier three-class pilot. Added paired-bundle loading checks without evaluating
  any test photos: both checkpoints load with the same 177-label order.
