# LeafCheck

LeafCheck trains and evaluates a custom CNN for 177 plant-condition categories
and runs saved models in a local image-upload comparison website. The main model
starts with randomly initialized weights. Standard PyTorch components supply
convolutions, automatic gradients and optimization; no pretrained weights are
used in the final CNN.

The comparison app in
`leafcheck-demo/` is tracked as ordinary source in this repository. It began as
an unconfigured gitlink with no remote; its earlier separate history and local
changes were preserved before consolidation. See `docs/repository_history.json`
and `artifacts/leafcheck-demo-history.bundle` for the recoverable original history.
The original metadata is retained under `.git/leafcheck-demo-history`; use the
parent repository for new commits. Datasets, checkpoints, caches,
local bundles and archives follow `.gitignore`; they are not large Git binaries.

## Status and evidence

The CSC-427 final project, **Plant Health Classification and an Interactive
Application**, delivers the 224 × 224 residual CNN from seed 434, epoch 53.
The model has 724,713 trainable parameters and 177 output logits. The frozen
scoring policy averages softmax distributions across eight orientations of
the same prepared photo.

Validation macro-F1 rose from 0.600999 at 128 pixels to 0.608911 at 160 pixels,
then to 0.621631 with mild brightness and contrast augmentation. The 224-pixel
pair reached 0.641060. The completed 256-pixel pair reached 0.644226, short of
the declared +0.005 promotion rule. Orientation averaging improved both 224
seeds by a mean of 0.007758 and met the CPU latency guard. The selected
checkpoint scored **0.649221 macro-F1** and **89.7541% accuracy** on the 16,836
validation images with the deployed scoring policy.

| Reserved test, same 8,425 images | Correct | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Selected custom CNN | 7,556 | 89.6855% | 0.616940 |
| Compact177 full-data reference | 7,234 | 85.8635% | 0.518430 |
| Most frequent training label | 731 | 8.6766% | 0.000902 |

On the separate 183-photo PlantDoc check, the CNN scored 26/183 correct
(14.2077%, three-label macro-F1 0.219846). The conditional frequency rule
scored 79/183 correct (43.1694%, macro-F1 0.201018). The CNN's higher macro-F1
comes with lower accuracy on this supplement. The primary test has 69 labels
with fewer than 10 images and 23 labels with zero observed recall.

Read `docs/final_model_selection.json` for the pre-test decision,
`docs/experiments.csv` for saved training validation results, and
`docs/final_model_report.md` for the recorded evaluation and figures.

- `docs/final_model_requirements.md`: verified requirements, assumptions and
  declared checkpoint/experiment selection rules.
- `docs/final_dataset_audit.md`: provenance, cleaning, group policy and limits.
- `data/manifests/leafcheck177_frozen.json`: immutable dataset fingerprint/order.
- `docs/comparison_protocol.json`: label coverage and exposure audit.
- `docs/model_defense_guide.md`: architecture, concepts and presentation questions.
- `docs/contributions.md`: actual roles, evidence and AI assistance.
- [Upload and blight study](docs/upload_blight_study.md): validation results by
  crop and source, app replay, experiment decisions and the upload-limit fix.
- [Field reliability follow-up](docs/field_reliability_study.md): new field
  evaluation, classifier-head experiments, and the app's uncertainty rule.
- [Crop-context and field-image follow-up](docs/potato_context_study.md):
  independently screened potato results, additional field training images,
  and the verified **Known crop** option.
- [Tomato transfer study](docs/field_optimization/tomato_transfer.md):
  matched scratch training, frozen PlantCity confirmation, and early- and
  late-blight error counts on new images.
- [Crop classifier study](docs/field_optimization/crop_head_study.md):
  two matched seed comparisons, the CCMT training release, and independent
  confirmation on Holeta potato and TLID tomato.

## Run the delivered study

Extract `artifacts/LeafCheck-CSC427-submission.zip` and open its
`START_HERE.md`. The report, defense deck, rehearsal guide, saved predictions
and source snapshot are included. The [GitHub repository](https://github.com/jacob-amey/ML-Final-Project)
has the published project history. Use the exact revision in the archive for
this delivery.

1. **Run the demo.** Extract the nested `LeafCheck-final-demo.zip`. On the
   tested Apple silicon Mac with Python 3.14 and Apple Swift tools, run the
   following from its `LeafCheck/` folder:

   ```sh
   xcrun --find swiftc
   python3.14 -m venv .venv
   .venv/bin/python -m pip install -r requirements-lock.txt
   chmod +x "Launch LeafCheck.command"
   ./Launch\ LeafCheck.command --port 8765
   ```

   If Swift is unavailable, install the Apple Command Line Tools with
   `xcode-select --install`. Use an unused port or `--port 0`. Open the local
   URL printed by the launcher, choose a JPEG or PNG and click **Run comparison**.
   The four-model comparison was verified on macOS. The bundle contains the
   weights and works without the research environment, notebooks or dataset.
2. **Recompute the saved scores.** From the extracted `source/` folder, run
   `python3.14 scripts/verify_saved_evaluation.py`. This uses only Python’s
   standard library and the archived membership, predictions and metrics.
   Install the research environment below if you want to execute the review
   notebook or run new inference on the packaged examples.
3. **Restore and retrain.** Obtain permitted access to the exact original
   master or frozen release. Follow the restore and verify commands in
   [the data guide](data/README.md), then build the caches and run the pinned
   commands below in a separate checkout. The local restore passed for all
   84,269 images. Individual upstream sources can be identified, but access
   and redistribution rights for the complete merged master are unconfirmed.

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
The submission archive includes the frozen membership and audit metadata.
See `data/README.md` for the tested restore command from the original master.

The cache command prepares only training/validation images and calculates
training-only normalization. Final evaluation prepares the test cache only after
checking the model/data freeze. Do not change class order, image orientation,
RGB conversion, interpolation, padding, scaling or normalization at deployment.

## Training and resume

To reproduce the paired 224- and 256-pixel training comparisons and the fair
compact reference, start in a fresh checkout with the verified frozen release
and research environment. Run the training commands sequentially on Apple MPS.
Each creates a separate run directory. Prepare the training and validation
caches first.

```sh
.venv/bin/python -m leafcheck.data --size 128 --workers 6
.venv/bin/python -m leafcheck.data --size 224 --workers 6
.venv/bin/python -m leafcheck.data --size 256 --workers 6
caffeinate -i .venv/bin/python -m leafcheck.train --config configs/plain177.json --run runs/final177/plain177_seed434
caffeinate -i .venv/bin/python -m leafcheck.train --config configs/residual177_224_seed434.json --run runs/final177/residual177_224_seed434
caffeinate -i .venv/bin/python -m leafcheck.train --config configs/residual177_224_seed435.json --run runs/final177/residual177_224_seed435
caffeinate -i .venv/bin/python -m leafcheck.train --config configs/residual177_256_seed434.json --run runs/final177/residual177_256_seed434
caffeinate -i .venv/bin/python -m leafcheck.train --config configs/residual177_256_seed435.json --run runs/final177/residual177_256_seed435
.venv/bin/python -m leafcheck.seed_analysis --runs runs/final177/residual177_224_seed434 runs/final177/residual177_224_seed435 --output docs/seed_stability_224_reproduction
.venv/bin/python -m leafcheck.compare_inference --pair docs/seed_stability_224_reproduction.json --output runs/final177/inference_comparison_reproduction
```

The saved evaluation uses the exact distributed checkpoint bytes. A training
reproduction can assess the same recipe and compare metrics, while backend
differences and checkpoint serialization can change file hashes.
The comparison command writes a new local `docs/inference_policy_comparison.json`
from those fresh runs. Keep the delivered records in their original checkout.

On this Mac, check whether the named detached session or an existing trainer
owns the job before resuming. The remaining development command acquires the
shared lock, preserves completed runs and resumes the last valid checkpoint.
It applies the declared paired decisions in order and records the candidate
for the final review while keeping the holdouts sealed.

```sh
/usr/bin/screen -ls
/usr/bin/screen -dmS leafcheck-final177 /bin/zsh scripts/run_remaining_development.sh
tail -f runs/final177/residual177_256_seed435_training.log
```

For an already completed native comparison, the same command reuses its valid
checkpoints. The follow-up driver records its work in
`runs/final177/final_development_tasks.log` and `docs/active_training.json`.

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
Exact interrupted-resume checks cover CPU and Apple MPS. Other accelerators,
including CUDA, need their own RNG-state handling and verification before an
equivalent resume claim; that portability work is outside this Mac execution.
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
  --run runs/final177/residual177_224_seed434 \
  --output artifacts/leafcheck177-v1 \
  --policy-record docs/inference_policy_comparison.json \
  --rationale "Selected by the paired validation protocol in docs/final_model_selection.json"
.venv/bin/python -m leafcheck.evaluate_final --artifact artifacts/leafcheck177-v1
.venv/bin/python -m leafcheck.evaluate_external --artifact artifacts/leafcheck177-v1
```

Final evaluation freezes and scores the completed Compact177 full-data reference
alongside the selected model and training-frequency rule. Compact177 uses the
same 177 labels and split membership; it is separate from the older three-class
pilots. The package command defaults to `runs/final177/plain177_seed434` for that
reference. Its saved SHA-256 is pinned in the selection plan. Restore the
distributed bundle to reproduce the reported results with the exact weights.

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

The final bundle includes `model.pt`, exact class mapping, model configuration,
preprocessing, the scoring policy, freeze record, hashes and saved evaluation.
It also includes the small inference runtime and
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

After both frozen evaluations finish, generate the report source and figures
before executing the review notebook. The notebook reads the saved development
comparison figure. This command does not retrain or change the checkpoint:

```sh
.venv/bin/python -m leafcheck.final_report --artifact artifacts/leafcheck177-v1
```

Then record both the pipeline and website suites and execute the review notebook:

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

After verification and the observed browser rehearsal, package the real local
application and test a fresh extracted process without dataset or notebook files:

```sh
.venv/bin/python scripts/package_demo.py
.venv/bin/python scripts/verify_demo_archive.py \
  --python /absolute/path/to/app-only-environment/bin/python
```

The latter interpreter should have only the app dependencies from
`leafcheck-demo/requirements-lock.txt`. The verification writes actual archive
hashes, startup and HTTP measurements; packaging alone is not a completed check.

## Interpretation and course submission

Softmax scores are model outputs, not calibrated probabilities of correctness.
The classifier cannot establish a definitive diagnosis, reliably detect every
unknown input or localize multiple plants/diseases. Source backgrounds, rare
classes and missing specimen IDs limit generalization. The
[field reliability study](docs/field_reliability_study.md) reports the separate
218-photo PLDD-UP check. The
[field expansion study](docs/field_expansion_study.md) documents additional
source-screened data and controlled experiments for real uploads.

The follow-up demo adds **Potato** under **Known crop** for photos you know
are potato. It compares the three supported potato conditions using the
frozen CNN. **Automatic** still covers all 177 labels. The updated demo archive
and launch instructions are in the [crop-context study](docs/potato_context_study.md).

Create ML's declared old training split contains 5,832 images now in the new test
set; its old validation split contains 1,683 more. Its predictions are available
for exploration, but scoring it on this holdout would be contaminated. Earlier
three-class pilots use different data and coverage. The final CNN, frozen
Compact177 full-data reference and declared simple rule share the valid
177-class held-out comparison.

The updated working proposal is in `docs/proposal.md`. The
[scope record](docs/scope_reconciliation.md) reconciles the earlier decks
and the 0.85 target with the measured result and delivered platform. The
[six-minute practice plan](docs/defense_rehearsal.md) covers the demo and Q&A.
The team remains
responsible for reviewing assisted work, following course policy, submitting the
required report/slides, and understanding the methods for Q&A. Do not attribute
assisted implementation, undocumented meetings or unverified author activity to
independent student work.

The local delivery is `artifacts/LeafCheck-CSC427-submission.zip`. Its
`START_HERE.md` points to the Word report, PowerPoint defense deck, standalone
demo archive, executed review notebook, source snapshot and saved evidence.
The original images remain in the local dataset and master folders. Use the
archive's `MANIFEST.json` and verification receipt to identify the files.
