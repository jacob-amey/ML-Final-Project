# LeafCheck model integration contract

The final model runs in the existing Python server on CPU. This uses the app's
current architecture, preserves Apple Create ML support and gives the browser
one consistent image-processing implementation. A separate browser export or
full new backend service is unnecessary for the local demonstration.

## Artifact and loading

The selected release is installed from `artifacts/leafcheck177-v1/` after its
selection/freeze and evaluations finish. The app's installer validates the files
and creates `leafcheck-demo/models/final_model.json`. That explicit reference
identifies a checkpoint-hash-named directory; final-model loading never chooses
the newest run automatically. Final evaluation files accompany the exact weights.

The bundle contains weights, architecture/runtime code, model configuration,
class mapping, preprocessing, package versions, dataset/selection fingerprints,
saved evaluation and example images. Training notebooks and the full dataset
are not needed for the packaged application. Large binaries remain outside Git.

## Input

The Python interface accepts a local image path. The website accepts JPEG,
JPEG-compatible MPO and PNG content, with a 10 MB upload and 24-megapixel limit.
MPO containers use frame zero. The class labels include leaves, some other plant
parts, healthy conditions and pests; one image produces one supported label.

Image preparation must match the bundle's `preprocessing.json` exactly:

1. Select frame zero and correct EXIF orientation.
2. Convert to three RGB channels.
3. Preserve aspect ratio with Pillow bilinear resizing and center gray padding
   `(128,128,128)` to the saved square resolution.
4. Represent pixels as float32 NCHW, divide by 255 once, then subtract the saved
   RGB mean and divide by saved RGB standard deviation. Statistics use training
   images only. Apply no inference augmentation.

Use the runtime's `Predictor` rather than implementing a second preprocessing
path in the frontend. The website's upload validation first converts the source
to an orientation-corrected lossless RGB PNG; the model applies its same shared
resize/scaling/standardization afterward.

## Output

The independent Python `Predictor.predict(path)` interface returns:

| Field | Meaning |
| --- | --- |
| `label` | Top class name from the saved 177-entry mapping |
| `score` | That class's softmax output, from 0 to 1 |
| `class_scores` | All 177 class names and scores, in saved class order; scores sum to one |
| `elapsed_ms` | This adapter's image preparation, network and output time |
| `model_version` | `leafcheck177-v1`; identify the exact weights with the artifact SHA-256 |
| `score_interpretation` | Explicit statement that scores are uncalibrated model outputs |

The existing `/api/predict` endpoint wraps predictions under
`models.final177`, alongside `cnn`, `starter300` and `createml`. Errors stay with
the affected model so available models can still return predictions. Upload
format/size errors produce an appropriate error response before inference.

Do not apply softmax twice, reorder the outputs, normalize the model's scores
over only the displayed top three, or interpret a high score as guaranteed
correctness. Unknown diseases, non-plant inputs, multiple plants and symptoms
with non-disease causes are not reliably detected or rejected.

## Shared responsibilities and agreement points

The model/data role supplies the artifact, mapping, preprocessing contract,
supported-label list, version, evidence, limitations and integration samples.
The application role handles selection/upload, request lifecycle and display.
Both should agree on supported file types, limits, raw label names versus display
names, uncertainty text, model-version display and the response/error schema.
Actual assisted work is recorded separately in `docs/contributions.md`.

The app displays saved final-model, Compact177 full-data reference and rule
scores for their shared 177-class test set, and separately labels the supplementary three-class PlantDoc check.
Earlier pilots have different label spaces and splits. Create ML's old training
and validation images overlap the new test, so its score is not a fair held-out
comparison on that set.

## Verification and performance

Integration checks compare all 177 API scores with standalone inference within
1e-5 absolute tolerance, verify saved example outputs, and cover RGB JPEG, EXIF,
grayscale/RGBA PNG, real MPO, invalid uploads and missing models. A fresh process
must start from packaged source and models without training files. A browser
rehearsal must upload a real image, display predictions and demonstrate a mistake.

Model size, loading time and warm batch-one CPU latency are measured separately
from training throughput. Full-request timing includes the other models and
upload handling and must not be presented as this CNN's forward-only speed.
See the finished artifact's `latency.json` and final verification evidence.
Compression/quantization are future options only if a measured bottleneck
justifies them and exported predictions/evaluation are checked again.
