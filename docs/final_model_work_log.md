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

## 2026-11-16 — training paused by user

- The user requested a pause after the system reported battery operation. Sent
  SIGINT to the verified training process; it stopped at a completed update.
  Saved state: 26 complete epochs, 74 completed batches of epoch 27, 9,472 images
  in that partial epoch and 12,060 total optimizer updates.
- Strict CPU model loading, finite tensors, optimizer/scheduler loading and
  Python/NumPy/CPU random-state restoration were verified. The MPS random-state
  payload is present. No training update was made during these checks.
- Last checkpoint SHA-256: `b7a9f7391dd141f636004f7a4d3a68f547cbb4f1afa20ff45adc6f43cd7dd026`.
  Backup: `artifacts/LeafCheck-paused160-20261116T145536Z.zip`; both checkpoint members were hash-verified.
- Closed the two temporary staging servers created for earlier verification.
  The final real-model browser rehearsal has not occurred. No test cache or
  frozen final evaluation was created. Do not auto-resume training; wait for
  an explicit user request. Exact continuation is recorded in
  `docs/training_pause.json` and below:

```sh
caffeinate -i .venv/bin/python -m leafcheck.train --run runs/final177/residual177_160_seed434 --resume
```

- Remaining: complete the resolution procedure and any justified paired repeat;
  freeze the selection plus Compact177 reference; evaluate the primary holdout,
  rule/reference and supplementary collection; install the real final artifact;
  execute full tests, notebook review, fresh browser rehearsal and delivery
  archive verification; record final results, documentation and commits.

## 2026-11-16 — authorized continuation on AC power

- The user requested continuation and removed time/resource intensity as a limiting
  preference. Verified AC power, ample memory, MPS availability, matching dataset
  fingerprint and the exact saved checkpoint hash before restarting.
- Found the working checkout on `dev` at 4874ab0. Preserved and rechecked all 37
  untracked report-draft files and the history bundle; no website source differed
  from the saved feature branch. Restored the model branch and merged the user's
  newer history-bundle commit, preserving it in 5160705. No remote push occurred.
- Resumed `residual177_160_seed434` from epoch 27, after 74 batches. The earlier pause
  record is historical; current state is `docs/active_training.json`. Both held-out
  sets remain unopened. Further trials require validation evidence and recorded
  hypotheses; no globally optimal-model claim is made.

- Completed the first 160-pixel procedure: 64 full epochs, selected 60, validation
  macro-F1 0.608147, accuracy 0.879069, loss 0.515985. Full reload verification
  passed. Total active procedure time 6,213.4 seconds includes the pre-pause work;
  paused wall time is excluded. The matched seed gain over 128 is 0.004466.
- The declared 160-pixel seed 435 repeat started after the first run completed.
  Mild illumination configurations are prospective; neither has run yet.
- Supplemental integrity review inspected the 36 closest cross-split representative
  pairs from the remaining 792 candidates, with enlarged native views for two
  visually similar rice pairs. Found no confirmed shared photograph; no release
  bytes, labels or splits changed. Remaining 756 candidates and unknown specimen/
  session IDs remain limitations. Raw integrity views did not include predictions
  and do not guide model choices; augmentation was declared before this review.

- While the second resolution seed was running, macOS detected a 45 W adapter
  but the battery continued falling. The user connected a higher-power charger;
  the system then reported 94 W and a rising charge. The exact training state
  continued without changes to data, weights, optimizer or schedule. Resource
  observations distinguish this power condition from algorithm comparisons.
- The runtime package now checks the selected run's architecture/preprocessing
  source hashes before freezing. The compact reference's earlier implicit
  float32 normalization and the explicit float32 implementation were checked
  to produce bitwise-identical tensors under its original default.

- Completed the 160-pixel seed repeat: early stopping after 51 full epochs,
  selected epoch 41; validation macro-F1 0.609675, accuracy 0.876990,
  unweighted loss 0.500699; 3,340.2 seconds. Full reload check passed.
- Paired 160-pixel mean macro-F1 is 0.608911 versus 0.600999 at 128, a
  gain of 0.007912. Both seeds improve; numeric configurations differ only
  in resolution and core training sources are identical. Training-only
  statistics were fitted separately by the same algorithm. Keep 160 as
  the development reference; two seeds are not significance.
- The preregistered mild-illumination trial started after the repeat
  completed. Only augmentation changes; all 59,008 training images remain.
  The final test and supplement remain sealed.

- Visually checked original and mild-illumination/D4 views for fixed training
  indices 1234, 15555 and 40000. Image content, outlines and visible markings
  remain present. Preview IDs, seed and SHA are recorded; this is a three-image
  software sanity check, not disease-evaluation evidence.
- In the historical resumed epoch 27, the logged maximum gradient norm covers
  the post-resume segment. Every update was still checked for finite gradients;
  weights, optimizer/RNG state and partial training metrics were restored.

- Expanded the pretest scaling plan after the user's removal of compute/time
  caps: one moderately wider channel configuration and one 224-pixel input
  check, separately controlled after regularization. Positive completed
  capacity/resolution evidence motivates these screens; larger is not assumed
  better. Fixed effect-size and seed-repeat conditions prevent an unbounded
  architecture grid. Preparing224 train/validation caches changes no split,
  image bytes or labels; no test cache is built.

- The configurable resource benchmark passed a CPU software smoke run using
  four real training images, a reduced-width177-output network, one warmup
  and two timed updates. Loss/gradient checks and an actual parameter change
  passed. This is not predictive-quality evidence or a larger-model MPS
  throughput result. Those representative MPS benchmarks remain pending.
- Clarified that the original Small CNN used120 training images and460
  validation images; its historical protocol differs from the177-class release.

- Completed the first mild-illumination procedure:51 full epochs, selected41;
  macro-F1 0.624215, accuracy 0.884830, loss 0.462634;3,459.5 seconds.
  Full reload verification passed. Matched seed434 change is +0.016068 F1,
  +0.005761 accuracy and -0.053352 unweighted validation loss.
- The independent illumination seed435 repeat started automatically only after
  completion. Recipe acceptance remains pending. The224 train/validation cache
  uses exactly the same row/label/group metadata, fitted training-only stats
  and no test pixels; this preparation ran concurrently with part of seed434.

- Completed the additional raw-file review of all366 cross-split candidates
  from the remaining792-row list. Inspected330 additional pairs, seven larger
  rice-source views and eight feature-correspondence cases. No additional
  shared photograph was confirmed. The isolated OpenCV correspondence check
  recovered113 inliers for a known cropped/resized/JPEG same-photo software
  control; sparse/repeated-point/background matches did not establish identity
  in the candidates. This is integrity software evidence, not CNN accuracy.
- No images, labels, groups or split membership changed. The426 remaining
  listed candidates are within splits; biological IDs, missed crops/edits and
  annotation quality remain limitations. Training dependencies were untouched;
  the separate integrity environment records its actual versions.

### 2026-11-17: resumed completed regularization evidence

Both illumination and stronger-decay seed pairs finished and reloaded correctly. Illumination mean macro-F10.621631 exceeds D4 mean0.608911. Stronger decay mean0.620684 does not improve the illumination reference; retain weight_decay0.0001. New width and224 screens hold other settings fixed and follow the already declared repeat/acceptance rule. Integrity feature evidence rerun reproduced its saved hash exactly. No test predictions inspected.

### Completed wider-channel screen

The32/64/128/256-channel160-pixel CNN early-stopped at45full epochs, selected35, validation macro-F1 0.623912. This does not exceed the matched illumination reference0.624215. The declared repeat trigger was not reached; no second seed or added width is accepted. This limited one-seed result does not rule out all wider architectures. Saved checkpoint reload passed. The independent224-pixel screen has started; no test scores inspected.

### Completed first224-pixel procedure

Early stopped after63 full epochs; selected53, validation macro-F1 0.643382, accuracy0.893799, loss0.431804. Reload verification passed. Matched seed434 improvement0.019167 exceeds the predeclared0.005169 trigger. Seed435 started with identical numeric settings except seed; acceptance awaits the paired result. Test and supplementary model predictions remain sealed.

### Conditional native-size check declared before results

The first224 result is promising while its second seed remains running. Training-only geometry shows36925/59008 images exactly256x256,56431 with short side>=256. If224 is accepted, screen native256 with one variable changed; repeat and accept only by the documented gain rule. Train/validation cache preparation is CPU-only and never reads the test split. Additional cache preparation overlaps part of the seed435 run, so elapsed times are not controlled method-cost comparisons.

The256 train/validation cache completed with exact224 row/ID/label/group metadata and59008/16836 counts. Training-only statistics are saved; no256 test cache exists. The second224 run later detected45W AC and declining battery; the user was asked to reconnect higher-power charging while authorized work continues.

The user reconnected higher-power charging; the Mac detected94W and subsequently80% battery charging. No source/data/algorithm change was needed. Latest filesystem seal check confirms no test cache or frozen final artifact; native256 cache preparation remains train/validation only.

### Paired224 resolution accepted

Both full procedures completed/reloaded: seed43463epochs selected53, F1 0.643382; seed43557epochs selected47, F1 0.638739. Mean0.641060 versus160 illumination0.621631 (+0.019429), both matched gains positive, observed range0.004643. Two seeds are limited stability evidence, not significance. The predeclared conditional native256 screen is now warranted; its repeat trigger is0.005 and paired acceptance gain0.005. No test predictions have informed these decisions.

The native256 sequential benchmark passed finite-loss/gradient/update checks:724713parameters,0.3184s/update versus2240.2419s/update, batch128. Post-update driver allocation4.31GB is an allocator snapshot, not a peak. Full256 training started from random seed434. Class-level224 paired changes are saved; eight classes have zero observed recall in both224 seeds versus seven in both160 illumination seeds, so gains are not universal.

### Remaining within-split raw identity review completed

Reviewed all426 remaining pairs on36 contact pages,15 larger source views and15 geometric cases. Verified raw hashes for444 distinct files. No additional shared photograph was confirmed; source release count,177 labels, groups, splits and fingerprint are unchanged. Degenerate34/20-inlier cases collapsed to one destination; the known same-photo software control recovered113inliers at104positions. Evidence is raw identity QA only, not model performance. All792 listed representatives now have supplemental review; undocumented biological identity and unlisted edits remain risks. Test predictions stay sealed.

### November 22 recovery and delivery preparation

The clean dev checkout had no live trainer. The seed 435 checkpoint recovered
at epoch 1, batch 100 with saved optimizer, scheduler and RNG state. The core
source hashes, configuration and frozen dataset fingerprint matched. Training
resumed on MPS behind a shared lock and completed its first full epoch. The
cached README-only collaborator commit remains in history, and the detailed
instructions were restored on the local feature branch. A launch service did
not start successfully. The monitored terminal session owns the live process.
The next development checks are queued behind its lock and require completed
paired evidence. See nov22_resume_record.json and the selection plan.

## 2026-11-22 — resumed comparison and final validation decision

- The detached session resumed the native 256-pixel repeat from epoch 5,
  batch 62 and finished after 58 full epochs. Best epoch 56 reloaded at
  0.637202 macro-F1, 89.1780% accuracy and 0.429230 loss. The matched seed 434
  improved by 0.007869 versus 224 pixels, and seed 435 dropped by 0.001537.
  The 0.003166 paired mean gain fell below 0.005. The conditional longer-budget
  trial did not qualify.
- The validation-only scoring trial used the same 16,836 images and two
  accepted 224-pixel checkpoints. Training-frequency logit adjustment had
  mixed per-seed changes and a mean gain of 0.000061. Averaging the softmax
  distributions over eight orientations raised F1 by 0.005840 and 0.009675,
  with CPU p95 of 42.56 ms over 16 warm calls. Seed 434 epoch 53 led with
  0.649221 macro-F1 and 15,111 of 16,836 correct. The comparison retained fixed
  weights and validation-only evidence.
- Both incumbent training curves stopped at the learning-rate floor. Sparse
  classes, residual source differences and persistent confusions remained after
  the paired tests. Choice of the 224-pixel model and eight-view policy was
  recorded in `final_model_selection.json` and committed in `1e65c57`. The
  artifact was frozen before building the test cache or predicting on either
  reserved collection.

## 2026-11-22 — frozen evaluation and delivery checks

- The selected artifact correctly classified 7,556 of 8,425 frozen test images
  with 0.616940 macro-F1. Compact177 scored 7,234 and 0.518430. The frequency rule
  scored 731 and 0.000902. Test predictions independently reconcile to the saved
  metric counts. The 183-photo PlantDoc supplement got 26 correct for the CNN
  and 79 for the rule, with macro-F1 of 0.219846 and 0.201018 respectively.
  These results informed documentation and examples only.
- The bundle installer validated the evaluated artifact and loaded its explicit
  SHA-256 reference. All 24 pipeline and website tests passed. The first review
  notebook execution surfaced an undeclared pandas dependency. The notebook
  now renders the tables with the standard library and IPython and ran to
  completion in the existing environment.
- The browser used the file chooser to rehearse the success and high-score
  mistake, and displayed a clear error for unreadable input. The fully
  extracted demo started via its launcher with the isolated app environment.
  All 88 files matched checksums. All four real models predicted on both photos.
  The frozen CNN had median 38.2 ms warm CPU inference. Four-model requests in
  the extracted app had median 52.2 ms on loopback.
- Restored all 84,269 files into a temporary folder from the original master
  and frozen membership records. The full release verifier passed. This
  validates the packaged metadata restoration workflow without changing the
  existing release.
- Rendered and reviewed all seven report pages and nine exported slides. The
  deck has an editable held-out comparison table. The delivery ZIP keeps data,
  saved weights and evidence local. The full master and prepared image files
  are available in the original local locations.

### Stop checkpoint verification

- The final 224-pixel seed-434 procedure has 63 complete epochs of 59,008
  images. The later 256-pixel seed-435 comparison finished 58 epochs. No
  training process or detached training session was active at inspection.
- Reloaded the frozen 177-class artifact on CPU and matched its weights to the
  selected epoch-53 checkpoint. Rechecked all 59 bundle file hashes and the
  submission archive hash. The deployed eight-view policy holds the recorded
  validation result of 0.649221 macro-F1. Its paired validation gain was
  0.007758. The 256-pixel recipe's paired gain of 0.003166 fell below its 0.005
  acceptance threshold.
- The project is stopped at `verified_delivery_complete` with the existing
  frozen evaluation and local submission archive. This check verified
  identities and completion records without generating new holdout predictions.

## 2026-11-24 — crop context and field follow-up

- The two independently initialized 224-pixel CNNs improved same-split
  validation macro-F1 from 0.649221 to 0.657068 when averaging their predictions.
  The 177-label course artifact remains the automatic app model.
- Two matched scratch runs per recipe tested a potato-only field specialist.
  The blend improved date-held Mduma validation, then trailed the original CNN
  given potato in a frozen check on 3,407 screened Okara images. That
  specialist stays experimental.
- The deployment adds optional potato context using the original checkpoint.
  Verified the complete upload path in a fresh archive extraction, including
  the crop selector, API parity, a success, a wrong condition and invalid
  input. Nine pipeline tests and 19 website tests passed.
- Added 578 screened TOM2024 field training images and completed both arms
  of the four-epoch replay. Neither arm qualified for replacement of the
  original CNN. See the [study and replay receipt](potato_context_study.md).
- Verified 508 additional reserved tomato images and the directional
  blight errors for both automatic models and their diagnostic known-crop
  variants. The automatic ensemble's master-validation gain did not
  generalize to the new tomato source.
- Verified PlantSeg exposure and built 722 reproducible training-only crops
  linked to their original parents. Completed both four-epoch arms. The
  selected crop checkpoint scored 0.639962 versus 0.640059 for the same-image
  control and 0.649221 for the original weight file. Both arms gave the same
  field-validation score. Kept the original model and the verified potato
  upload option.

## 2026-11-25 — field first optimization

- Live browser upload displayed Tomato / Healthy for the packaged success.
  An unreadable file produced the expected error. Standalone and local
  upload agreed exactly on all 177 scores for both packaged fixtures.
  [Runtime and results](field_optimization/upload_baseline.json)
- Tested crop consensus before fitting more weights. The full validation
  reference reproduced 0.649221 macro-F1. The crop-first candidate gained
  0.002995 mean joint F1 on the grouped Sheikh and Mduma development data,
  below the registered 0.020 gain, and lost 0.001826 master F1. The softer
  reweighting had zero field F1 gain. Both were rejected.
  [Protocol](../configs/leafcheck_crop_decoding_v1.json) ·
  [Outcome and per-class evidence](field_optimization/crop_decoding.json)
- Acquired 158 raw camera-dated Pakistan tomato photos under CC BY 4.0.
  The release retained 142 examples in one development group. Its frozen
  automatic baseline got 7 correct, joint macro-F1 0.099783, and named a
  different crop on 114. The existing display rule showed 12 predictions,
  all wrong. With the reference crop supplied, the condition diagnostic
  got 27 correct and 0.177608 macro-F1. This is evidence for the need to
  improve condition recognition alongside crop identification.
  [Baseline and directional errors](field_optimization/pakistan_baseline.json)
- The next model comparison should isolate the effect of field training on
  a randomly initialized condition learner, with the same class coverage,
  dose-matched control, and both the supplied-crop and joint prediction
  metrics. Keep the 142-image capture group in development, check at least
  one other independent source, and guard broad 177-label performance.
  Candidate training already has 11,828 master tomato images, and screened
  tomato supplements of 1,127, 419 and 328 images. The supplements cover
  early blight, late blight, healthy, bacterial spot and Septoria. They
  leave leaf mold and yellow leaf curl dependent on the master.
  Pota-Toma-To was screened from its pinned archive directory. Its 435
  raw files are grouped under Google Search paths and have matching
  processed families. The available inventory does not establish
  independent field capture or original asset rights, so it stays out
  of the experiment. [Publisher record](https://data.mendeley.com/datasets/354fsxwccb/1)
- Updated the editable report and checked its format against the available
  requirements. The current report and original course archive are separate
  artifacts. [Requirements checklist](final_model_requirements.md#field-update-report-checklist)

### Matched tomato training and independent confirmation

- Completed and reloaded four scratch-trained tomato models. The field
  arm reached 0.315641 mean known-tomato development macro-F1 against
  0.279810 for matched master replay. Both field seeds missed the
  master-validation guard. Joint automatic routing lost 0.016558
  mean field macro-F1 in both seeds.
- The 50:50 known-tomato score blend passed development at 0.317404
  mean macro-F1 and was frozen before inference on the independent
  PlantCity release. On the same 134 originals given tomato, it got
  34 correct and 0.233403 macro-F1 against 44 correct and 0.298658 for
  the original. The original automatic model got 26 of 299
  across six crops. Wrong crops accounted for 224 of the 273 errors.
  The blend failed confirmation and the app kept the course checkpoint.
- Recomputed all three results and class supports from the saved rows.
  The reconciled PlantCity evidence is now locked to the completed
  follow-up. See [methods, directional errors and reproduction](field_optimization/tomato_transfer.md).
- Ran all nine training-pipeline tests and all 19 website tests with a
  freshly started local service. Rendered and inspected the editable
  ten-page field study report. The original combined course submission
  archive remains separate. [Report verification](field_optimization/report_tomato_verification.json)

### Crop classification and independent source confirmation

- Registered and restored 529 screened CCMT crop-only photos, 154 tomato,
  217 corn and 158 cassava. Verified the original bytes against the
  publisher hashes, visually reviewed one example per source folder,
  and reproduced the training-only release.
- The two-seed matched crop study completed all six epochs per arm.
  Joint mean field development macro-F1 rose from 0.050016 to 0.091941
  and 0.083875. Most of the gain came from the Mduma potato group.
  Master validation macro-F1 was 0.646089 and 0.647168, down from
  0.649221 within the development guard. A metadata fix recovered six
  completed control epochs with their exact weights.
- Froze seed 434 before evaluating 245 independent Holeta and TLID
  photos. The candidate gained 0.006858 mean macro-F1 and reduced
  wrong-crop rate by 4.48 points. Both differences missed the declared
  +0.020 and 5-point thresholds. Kept the existing app checkpoint.
  [Methods, metrics and reproduction](field_optimization/crop_head_study.md)
- Verified an initial Image Safari intake of eight public original
  image/sidecar pairs across five crop folders. The source has explicit
  site and capture-date paths, and its file stems differ from the byte
  hashes. This remains intake work with zero inference. Disease-label
  compatibility requires separate evidence.
  [Publisher record](https://registry.opendata.aws/artemis-image-safari/)
- Completed the fixed central-view evaluation. Mean field development
  macro-F1 changed by −0.000534 and master development macro-F1 by
  −0.001225. Top-1 accuracy on the field groups did not improve.
  Kept original full-image inference. No test files were used.
  [Protocol](../configs/leafcheck_center_view_v1.json) ·
  [Evidence](field_optimization/center_view.json)
- Admitted 275 Image Safari photos across seven crops after metadata,
  byte, grouping and prior-overlap checks. Four whole sites supply 129
  training photos. Six held sites supply 146 crop-only validation views.
  The original automatic model identified 18 crops correctly. Summed
  original crop probabilities identified 21. The rejected field crop head
  identified 15 alone and 20 in the mixed policy.
- Reviewed all 146 validation thumbnails for framing without viewing
  per-photo predictions. The original got 10 of 78 primarily foreground,
  eight of 40 whole-plant or plot views, and zero of 28 sparse views.
  The crop/site composition differs by framing. Treat these as diagnostic
  counts and preserve the full cohort. Disease labels are unavailable.
  [Crop metrics and framing rubric](field_optimization/imagesafari_baseline.json)
- Completed a matched Image Safari scratch pilot with 129 new unique
  training views and six occurrences per view per epoch, matched
  by crop in the control. Both arms completed six epochs and reloaded
  the selected checkpoints. The site-held crop score rose from
  0.150225 to 0.164976 macro-F1, a 0.014751 gain over control.
  The target of 0.030 was missed. The registered rule stopped the trial
  after the first seed. The 275-source release reproduced its hashes
  and site assignments in a separate build. [Evidence](field_optimization/imagesafari_pilot.json)
- Replayed the original bytes of three labeled PlantCity late-blight
  photos through the four-model local upload API. The frozen model
  produced one late-blight result, one early-blight result and one
  wrong-crop result, cherry leaf spot. All 177 scores matched standalone
  exactly for every upload. These cases demonstrate model errors on
  publisher-labeled examples. The team's original photos are unverified.
  [Upload replay](field_optimization/plantcity_upload_replay.json)
- Screened additional tomato sources before admitting photos. One
  candidate archive contained the same zucchini image in three disease
  folders. A greenhouse late-blight repository contained 1,911 exact
  matches to included master images, including 209 from the course
  holdout. Neither release was used for condition fitting or evaluation.
  [Catalog disposition and byte evidence](field_optimization/next_source_screen.json)

### Evening continuation — explicit tomato context

- Verified all 473 bytes against the Hossain catalog and retained 385
  photos after screening repeats, label conflicts, and possible overlap.
  All records share one EXIF capture day. The reviewed sample showed
  detached leaves on pale surfaces. The whole source was held together
  and frozen before scoring.
- Reconstructed both outcomes from the exact prediction records.
  The 177-label CNN got 0/385 right and named another crop 312 times.
  Explicit tomato context got 15/385 right, improving macro-F1 by 0.050251
  and accuracy by 3.896 points. Healthy and late-blight recall stayed
  zero. The accuracy gain missed the five-point gate, so the app stays
  with the frozen 177-label checkpoint and existing potato option.
  [Study](field_optimization/tomato_transfer.md#explicit-tomato-context-on-a-new-source)

### Matched crop loss and unsupported uploads

- Completed both six-epoch scratch runs with 59,008 training images
  per epoch. The candidate improved field macro-F1 by 0.017231 and
  reduced wrong-crop rate by 2.64 points relative to the matched control.
  Both gains missed the registered advancement criteria.
- Restored and screened 48 Open Images photos for development.
  The current threshold suggested labels on 13 of the 45 admitted
  unsupported photos. Fixed probability or raw-logit signals provided
  insufficient separation.
- A source-held linear probe improved mean AUROC from 0.570 to 0.770
  for nonplants and from 0.461 to 0.622 for unsupported plants. The
  unsupported-plant result missed 0.65 and the result varied by source.
  Fold scores reproduced exactly and were recalculated from the
  prediction records. Kept the existing checkpoint and upload policy.
- Verified normal browser uploads in a fresh extraction of the updated
  demo. Confirmed the 43-crop list, score explanation, healthy tomato
  fixture and high-scoring typewriter mistake. Separate HTTP checks
  verified model readiness and both invalid-image responses.
- Rendered and inspected the ten-page editable Rev4 report. The original
  frozen course archive remains separately available.
  [Methods and delivery evidence](field_optimization/field_v8_study.md)

### Field intake and partial-label potato experiment

- Screened the 3,477-image GVLiD release and found 1,344 folder/sidecar
  label disagreements, two exact decoded groups spanning folders and
  conflicting date evidence. The archive stays outside the study.
- Verified 305 public IIT Mandi field potato photos and screened them against
  prior releases. The source paper documents both blights. Its published
  labels allow an early-or-late target for 217 images and a healthy target
  for 88 images. The full set entered the treatment arm only.
- Ran two matched custom scratch CNNs to 18 epochs. Both selected
  checkpoints gave 191/211 correct on the date-held Mduma validation.
  The treatment gained 0.001316 macro-F1 while master potato validation
  declined by 0.057221. The prespecified advancement gate failed. The
  frozen CNN, its inference policy and the optional potato route stayed
  in service. Recomputed all selected metrics from confusion counts
  and verified model reload and exact training membership.
- Updated and rendered the ten-page editable Rev5 report with the matched
  potato result. [Data and model evidence](field_optimization/field_v9_intake.md)

### Corn field validation and source matching

- Expanded the Pretoria intake to 96 publisher originals from four
  compatible single-label folders. Removed two possible repeat views.
  The 94 date-recorded photos remain in development. Automatic mode
  got 21 right, selected the wrong crop 22 times, and made 51 condition
  errors with the right crop. Supplying corn gave 26 correct. The separate
  inherited generic rust class accounted for eight common-rust errors.
- Preserved the frozen labels and confirmed the source support of the two
  rust names independently of the test results. Recorded the 94 verified
  predictions and four-label scores in the updated, ten-page report.
  [Methods](field_optimization/field_v9_intake.md#expanded-corn-field-validation)
- Registered a matched, label-preserving intake and comparison using
  additional RAW_Image JPEGs from Bangladesh. The other public folders
  contain derivative views or labels without an exact mapping. The
  full 177-label deployment remains independent of this diagnostic.

### Seasonal corn data comparison

- Verified 429 original archive images and admitted 412 from the seasonal
  corn source after overlap screening. Photos remained training-only.
  Common-rust filenames were concentrated on one capture day.
- Completed the matched scratch trial. On Pretoria, the treatment
  scored 22/94 and 0.221419 macro-F1 against 26/94 and 0.238126 for the
  control. It correctly identified two common-rust cases, and recall
  for northern leaf blight fell from 14/22 to 7/22. Both runs reloaded
  their selected checkpoints and missed the master guard. The frozen
  177-label deployment stayed in service.
- Added the result to the editable ten-page report and verified its
  rendered pages. [Evidence](field_optimization/corn_seasonal_study.json) ·
  [Report verification](field_optimization/report_seasonal_verification.json)
