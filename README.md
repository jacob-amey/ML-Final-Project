# LeafCheck

LeafCheck trains and evaluates a custom CNN for 177 plant-condition categories
and runs saved models in a local image-upload comparison website. The main model
starts with randomly initialized weights. Standard PyTorch components supply
convolutions, automatic gradients and optimization; no pretrained weights are
used in the final CNN.

The current task branch is `codex/final-leafcheck177`. The comparison app in
`leafcheck-demo/` is tracked as ordinary source in this repository. It began as
an unconfigured gitlink with no remote; its earlier separate history and local
changes were preserved before consolidation. See `docs/repository_history.json`
and `artifacts/leafcheck-demo-history.bundle` for the recoverable original history.
The original metadata is retained under `.git/leafcheck-demo-history`; use the
parent repository for new commits. Datasets, checkpoints, caches,
local bundles and archives follow `.gitignore`; they are not large Git binaries.

## Status and evidence

Full-data training and validation experiments are in progress at the user's
request. The paired 128/160-pixel comparison is complete; 160 pixels improved
validation macro-F1 in both seeds. The declared augmentation and regularization
checks follow that reference. `docs/active_training.json` identifies the current
run and its live status file. The historical pause record is retained separately.

Final selection, held-out evaluation and the real final-model website rehearsal
remain pending. No final-model predictions have been inspected on the primary
holdout or supplementary collection. Read `docs/experiments.csv` for saved
validation results and `docs/final_model_work_log.md` for completed work.

- `docs/final_model_requirements.md`: verified requirements, assumptions and
  declared checkpoint/experiment selection rules.
- `docs/final_dataset_audit.md`: provenance, cleaning, group policy and limits.
- `data/manifests/leafcheck177_frozen.json`: immutable dataset fingerprint/order.
- `docs/comparison_protocol.json`: label coverage and exposure audit.
- `docs/model_defense_guide.md`: architecture, concepts and presentation questions.
- `docs/contributions.md`: actual roles, evidence and AI assistance.

## Environment and dataset

Use the existing project `.venv` where possible. The verified environment is
Python 3.14.7, PyTorch 2.15.0, torchvision 0.29.2, Pillow 13.0.0 and Apple MPS.
Training also uses NumPy; figures use matplotlib. Exact run environments and
source fingerprints are recorded in each run directory.
`requirements-lock.txt` records the exact 108 installed package versions used on
this Mac; it does not recreate macOS or its accelerator driver.

For a new checkout with a compatible Python 3.14 installation and no `.venv`,
create the research environment first:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

Keep an existing working environment intact. The website has a smaller separate
dependency list and tested lock under `leafcheck-demo/`; it needs no notebook or
plotting packages. The verified isolated app environment is recorded in
`docs/minimal_runtime_environment.json`.

From this project folder:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/prepare_final_dataset.py verify --workers 4
.venv/bin/python -m leafcheck.data --size 128 --workers 6
.venv/bin/python -m leafcheck.diagnose --size 128
.venv/bin/python -m unittest discover -s tests -v
```

The prepared release is `data/processed/leafcheck177_final/`: 59,008 training,
16,836 validation and 8,425 test images across all 177 labels. Every image has a
traceable source/content ID. Known related groups remain within one split; many
physical specimen identities are unknown. Some classes have only six images.
The local master is preserved. See `data/README.md` for preparation/restoration
evidence and source records. A checkout does not download the image data.
Reproducing training requires the exact provided master or prepared release,
including the recorded metadata and bytes. This repository does not establish
reuse rights or a public download location for every upstream master image.

The cache command prepares only training/validation images and calculates
training-only normalization. Final evaluation prepares the test cache only after
checking the model/data freeze. Do not change class order, image orientation,
RGB conversion, interpolation, padding, scaling or normalization at deployment.

## Training and resume

Each new run starts with fresh weights unless an explicitly documented later
refinement configuration says otherwise. The training entry point accepts no
test-split option. Full epochs visit every training image once, with whole-image
training augmentation. Validation uses deterministic preprocessing.

```sh
caffeinate -i .venv/bin/python -m leafcheck.train \
  --config configs/plain177.json --run runs/final177/plain177_seed434
caffeinate -i .venv/bin/python -m leafcheck.train \
  --config configs/residual177.json --run runs/final177/residual177_seed434
```

Run accelerator experiments sequentially. On a non-Mac, omit `caffeinate` and
choose an available device in the configuration. A run directory is unique and
cannot be silently overwritten. To resume an interrupted run:

```sh
caffeinate -i .venv/bin/python -m leafcheck.train \
  --run runs/final177/residual177_seed434 --resume
```

`last.pt` saves state at completed-update boundaries every 100 batches, after
each epoch and on handled interruption. `best_model.pt` stores the checkpoint
selected by validation macro-F1, with lower validation loss as the tie-breaker.
The checkpoint includes its architecture, class mapping and preprocessing.
Seeds, model/optimizer/scheduler states, RNG states and partial-epoch totals are
preserved for resume. Backend/device/version differences can still affect numbers.
Resuming a completed procedure preserves it and exits. To reproduce an experiment
again, choose a new empty run name. Restoring a recorded source snapshot means
placing its `leafcheck` package into a clean project checkout with the matching
data layout, not executing the snapshot directly from its archived subfolder.

The controlled weighting and duration/stability checks use these commands. They
are reproduction instructions, not prerequisites for running the saved demo:

```sh
caffeinate -i .venv/bin/python -m leafcheck.train \
  --config configs/residual177_weighted.json --run runs/final177/residual177_weighted_seed434
.venv/bin/python -m leafcheck.extend_run \
  --parent runs/final177/residual177_weighted_seed434 \
  --output runs/final177/residual177_weighted_long64_seed434 --max-epochs 64
caffeinate -i .venv/bin/python -m leafcheck.train \
  --run runs/final177/residual177_weighted_long64_seed434 --resume
caffeinate -i .venv/bin/python -m leafcheck.train \
  --config configs/residual177_weighted_seed435.json --run runs/final177/residual177_weighted_seed435
```

The continuation preserves the parent and its exact optimizer/scheduler/RNG
state. Its copied early epochs are not another independent fresh run. Incremental
runtime is recorded separately to prevent double-counting. The seed-repeat starts
with new random weights and a different shuffle/augmentation stream.

Inspect `history.csv`, `validation_metrics.json`, `validation_predictions.csv`,
`status.json`, configuration, environment and source snapshots under each run.
Generate curves and refresh the experiment register with:

```sh
.venv/bin/python -m leafcheck.report runs/final177/residual177_seed434
.venv/bin/python -m leafcheck.experiments
```

## Freeze and final evaluation

Choose using validation only. Record the experiment decision and stopping reason,
then freeze an explicitly selected **completed** run. The following shows the
command contract; use the actual selected run in the final manifest.

```sh
.venv/bin/python -m leafcheck.package \
  --run runs/final177/SELECTED_RUN \
  --output artifacts/leafcheck177-v1 \
  --rationale "Document the validation-based selection and stopping decision"
.venv/bin/python -m leafcheck.evaluate_final --artifact artifacts/leafcheck177-v1
.venv/bin/python -m leafcheck.evaluate_external --artifact artifacts/leafcheck177-v1
```

Final evaluation freezes and scores the completed Compact177 full-data reference
alongside the selected model and training-frequency rule. Compact177 uses the
same 177 labels and split membership; it is separate from the older three-class
pilots. The package command defaults to `runs/final177/plain177_seed434` for that
reference; use `--reference-run` only to name its reproduced completed run.

Final evaluation saves per-image predictions, correct/total, accuracy, macro-F1,
per-class metrics and support, confusion tables/figures, successes/failures,
the most-common-training-class rule and latency measurements. After seeing
test results, further proposed tuning belongs in future work. Rechecking an
unchanged frozen artifact is acceptable; do not select another model on this test.

The preregistered supplementary PlantDoc collection contains 183 screened photos
covering three tomato labels, with 57 overlaps/related photos omitted. Its own
manifest and protocol are under `data/manifests/leafcheck177_external*`. It is
evaluated only after selection, retaining all 177 logits as possible predictions.
Its macro-F1 averages the three represented labels; it must not be compared
numerically with macro-F1 over the full 177-class test. PlantDoc also appears
elsewhere in the master, so this is not an unseen-source benchmark. Generic
`Tomato leaf` is an inherited healthy proxy, and undetected related views may remain.

## Standalone inference and website

The final bundle includes `model.pt`, exact mapping/config/preprocessing, freeze
record, artifact hashes, saved evaluation and its small inference runtime. It
does not require notebooks or the full training dataset.

```sh
.venv/bin/python -m leafcheck.inference \
  --artifact artifacts/leafcheck177-v1 /absolute/path/to/photo.jpg
.venv/bin/python leafcheck-demo/backend/bundle_final.py \
  --artifact artifacts/leafcheck177-v1
.venv/bin/python leafcheck-demo/backend/server.py --port 8765 --open-browser
```

The installer validates all bundle hashes and updates the explicit
`leafcheck-demo/models/final_model.json` reference. It never guesses the newest
final checkpoint. Restart the app after changing the reference. You can also
double-click `leafcheck-demo/Launch LeafCheck.command`. Close the launcher with
Control-C when finished. The app's README covers standalone setup and earlier
model bundles. `LEAFCHECK_FINAL_ARTIFACT` can explicitly override the final bundle.

With the real bundled model and server running:

```sh
LEAFCHECK_TEST_URL=http://127.0.0.1:8765 \
  .venv/bin/python -m unittest discover -s leafcheck-demo/tests -v
```

These checks include earlier-model regressions, standalone-copy startup,
numerical final-model/API parity, orientation/color formats, invalid uploads and
private-file routing. Full historical-reference tests use the original research
environment; the running application needs no training data or notebooks.

To record both the pipeline and website suites and execute the review notebook:

```sh
.venv/bin/python scripts/verify_final.py --url http://127.0.0.1:8765
```

The command requires the real completed bundle, checks its identity against the
live server, saves actual test logs/results, and preserves an executed review
notebook under `runs/final177/verification/`. It does not retrain or select a model.
Open `cnn_final_model_review.ipynb` in VS Code or Jupyter with the **Python
(LeafCheck)** kernel for a guided architecture/evidence review. If that kernel is
missing in a new research checkout, register its environment with:

```sh
.venv/bin/python -m ipykernel install --user --name leafcheck --display-name "Python (LeafCheck)"
```

## Interpretation and course submission

Softmax scores are model outputs, not calibrated probabilities of correctness.
The classifier cannot establish a definitive diagnosis, reliably detect every
unknown input or localize multiple plants/diseases. Source backgrounds, rare
classes and missing specimen IDs limit generalization. No new independent field
benchmark is claimed.

Create ML's declared old training split contains 5,832 images now in the new test
set; its old validation split contains 1,683 more. Its predictions are available
for exploration, but scoring it on this holdout would be contaminated. Earlier
three-class pilots use different data and coverage. The final CNN and declared
simple rule share the valid full-177 held-out comparison.

The updated working proposal is in `docs/proposal.md`. The team remains
responsible for reviewing assisted work, following course policy, submitting the
required report/slides, and understanding the methods for Q&A. Do not attribute
assisted implementation, undocumented meetings or unverified author activity to
independent student work.
