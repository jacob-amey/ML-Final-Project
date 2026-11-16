# LeafCheck — local Mac app

LeafCheck compares the final 177-class custom CNN, Small CNN, Starter 300 CNN,
and Apple Create ML model on the same photo. The interface is ordinary HTML, CSS, and JavaScript. A small Python
server runs the models on your Mac and serves the interface to your browser.

**No ChatGPT account, hosting service, API key, Node.js, npm, frontend framework,
or build step is needed. After dependencies are installed, use is offline.**

## Start the app

1. Open this folder in Finder.
2. Double-click **Launch LeafCheck.command**.
3. Wait for the saved models to load. The launcher opens your normal browser at
   **http://127.0.0.1:8765**.
4. Choose a JPEG or PNG photo, or one of the included examples, then click
   **Run comparison**.

Keep the terminal window open while using the app. Press **Control-C** in that
window to stop it. Double-click the launcher to restart. If this copy is already
running, launching it again simply opens its browser page.

Your current project already has the Python environment and models needed to run
it. The launcher automatically finds that environment. It also supports a `.venv`
inside this app folder or an installed `python3` with the required libraries.

Open the local address in Safari, Chrome, Firefox, or your preferred browser.
Opening `web/index.html` directly does not start the model server; use the launcher.

## What is inside

```text
leafcheck-demo/
├── Launch LeafCheck.command    # One-click Mac launcher
├── requirements.txt            # PyTorch and Pillow only
├── web/
│   ├── index.html              # App interface
│   ├── styles.css              # Plain CSS; system fonts
│   ├── app.js                  # Uploads, live results, export
│   └── assets/                 # Icon and three example photos
├── backend/
│   ├── server.py               # Python standard-library HTTP server
│   ├── models.py               # Registers the four classifiers
│   ├── cnn.py                  # The notebooks’ exact PlantCNN architecture
│   ├── coreml_runner.swift     # Native Apple Core ML / Vision helper
│   ├── update_models.py        # Refreshes earlier pilot/Create ML bundles
│   ├── final_model.py          # Validates and loads the explicit final bundle
│   └── bundle_final.py         # Installs one explicitly selected final artifact
└── models/                     # Bundled trained weights and saved metrics
```

Python serves only the `web` directory. The server listens on your own Mac's
loopback address. Uploaded images are held temporarily for inference and removed
afterward. Results remain in the browser until you clear or close the page; the
export button downloads a JSON file when you choose to save a comparison.

All page resources are local, including fonts and examples. Every comparison runs
the real saved models. There is no cached-prediction mode or external image service.

## One-time setup on another Mac

Copy the **whole folder**, including `models/`. Model binaries are ignored by Git,
so a source checkout alone does not include them.

Use a Python installation compatible with the pinned packages, then open a terminal
in this folder and run:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The Create ML model uses macOS's Core ML and Vision frameworks. Apple Command Line
Tools are needed to compile the small Swift helper on its first launch. If missing,
install them with `xcode-select --install`. The compiled helper is reused in
`.runtime/`. These setup steps can need internet; normal app use does not.

## Run from a terminal

From the original `ML-Final-Project` folder:

```sh
.venv/bin/python leafcheck-demo/backend/server.py --open-browser
```

If using an app-local environment, run from this app folder instead:

```sh
.venv/bin/python backend/server.py --open-browser
```

Use `--port 8766` if another app is using port 8765. The server prints its address.

## Update the saved models after training

The app includes copies of the trained models, so the notebooks and datasets are
not required to run it. To refresh those copies after a new notebook run, execute
this from `ML-Final-Project`, then restart the app:

```sh
.venv/bin/python leafcheck-demo/backend/update_models.py
```

This refreshes only the earlier models: it copies the newest `best_model.pt` from each of `runs/leafcheck_small/` and
`runs/leafcheck_starter300/`, its saved validation/test metrics, and the exported
`PlantDataML.mlproj/Models/PlantDataML 1.mlmodel` into `models/`.

Install the selected final model separately, after its training, validation-based
selection, primary test evaluation and registered supplementary evaluation finish:

```sh
.venv/bin/python leafcheck-demo/backend/bundle_final.py \
  --artifact artifacts/leafcheck177-v1
```

This verifies the bundle and creates an explicit `models/final_model.json`
reference to a checkpoint-hash-named version directory. It never selects a newest
final run automatically. The bundle includes architecture/runtime code, weights,
class mapping, preprocessing, configuration, frozen evaluation and demo photos.
Restart the server after installation. The installer also copies the selected
held-out success and mistake into the local example area, with provenance metadata.

You can also override a model path before starting the server:

```sh
export LEAFCHECK_FINAL_ARTIFACT="/absolute/path/to/frozen/artifact/folder"
export LEAFCHECK_CNN_CHECKPOINT="/absolute/path/to/small/best_model.pt"
export LEAFCHECK_STARTER300_CHECKPOINT="/absolute/path/to/starter300/best_model.pt"
export LEAFCHECK_COREML_MODEL="/absolute/path/to/classifier.mlmodel"
```

## Reading results

- **Final Custom CNN:** custom architecture trained from random initialization on
  the full 59,008-image training split; all 177 labels. Selection uses validation
  macro-F1. Its saved 8,425-image test result appears beside the simple rule,
  with a separately labelled supplementary 183-photo, three-tomato-class check.
- **Small CNN:** original 120-image pilot; 3 tomato conditions.
- **Starter 300 CNN:** second experiment; 210 training, 60 validation, and 30 test
  images from the 300-image dataset; 3 tomato conditions.
- **Apple Create ML:** 177 plant classes; may predict other plants or conditions.

The two pilots restore their exact 128 × 128 RGB, gray-padding and pixel-scaling
pipeline from the notebooks. The final CNN uses its saved resolution, frame-zero
selection, EXIF correction, RGB conversion, bilinear gray padding, pixel scaling
and training-only channel standardization through one shared inference runtime. The
Core ML helper uses orientation correction and Vision center cropping and scaling.

Scores are model outputs, not calibrated accuracy estimates. Agreement is context,
not proof that a prediction is correct. The earlier experiments use different evaluation sets. The new test includes
5,832 images from Create ML’s declared old training split and 1,683 from its old
validation split. Create ML is not scored on this holdout. Only the selected final
CNN and declared simple rule share the reported 177-class test protocol. Timing includes different work in the CNN and Vision paths and is not
a speed benchmark. The model-details dialog reads saved metrics; the app never
trains a model or evaluates another held-out test set.

## Verification

With the app running, the original research environment can run:

```sh
.venv/bin/python -m unittest discover -s leafcheck-demo/tests -v
```

The full suite uses the original project’s notebooks, validation data, and
torchvision to independently verify saved reference predictions. Other checks
cover all four real models, invalid uploads, local-only browser dependencies,
and running a copy of the app without the training data or notebooks.


The final integration checks compare all 177 API scores with standalone inference
on real bundled photos, within an absolute tolerance of 1e-5. They also check
saved evaluation examples, orientation, grayscale/RGBA PNG and missing artifacts.
Synthetic color-format fixtures are software checks, not disease-evaluation data.
The full application can start from its packaged folder without notebooks or the
training dataset; copying the model bundle is required because Git excludes it.
