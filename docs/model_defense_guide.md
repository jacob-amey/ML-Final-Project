# Understanding and defending the final LeafCheck model

Use this guide alongside the actual selected configuration and results. The
implementation was assisted by Codex at the user's request. Explain what you
understand and verified; do not claim to have independently authored assisted code.

## What is custom and what the framework supplies

`leafcheck/model.py` defines the architecture and forward computation. Each new
constructor initializes fresh convolution and linear weights. There is no
pretrained classifier or feature extractor. PyTorch supplies layer mathematics,
automatic gradients and AdamW. “From scratch” here means designing/implementing
the architecture and training new weights, not manually programming convolution
and backpropagation from numerical arrays. This follows the user's stated
definition; the pasted course rubric does not impose a more specific definition.

Residual connections are an established idea, not a new invention of this
project. A block learns a correction to its input; the shortcut provides a shorter
gradient path. A 1×1 convolution projects the shortcut when channel count or
spatial size changes. The project implements its own compact arrangement using
this idea [1], rather than importing pretrained ResNet weights.

## Architectures and tensor shapes

For the 128-pixel experiments, `B` is batch size and tensor layout is batch,
channels, height, width. Both models use widths 24, 48, 96 and 192.

| Stage | Compact reference | Residual model |
| --- | --- | --- |
| Input | B×3×128×128 | B×3×128×128 |
| First downsampling | 3×3 convolution, stride 2 → B×24×64×64 | Same stem → B×24×64×64 |
| Feature stages | Three additional stride-2 convolutions → B×48×32×32, B×96×16×16, B×192×8×8 | Four two-convolution residual blocks → B×24×64×64, B×48×32×32, B×96×16×16, B×192×8×8 |
| Normalization and activation | Batch normalization and ReLU | Batch normalization and ReLU |
| Spatial aggregation | Global average pooling → B×192 | Global average pooling → B×192 |
| Classifier | Dropout, linear 192→177 | Dropout, linear 192→177 |
| Outputs | B×177 raw logits | B×177 raw logits |
| Trainable parameters | 253,257 | 724,713 |

A convolution's output width is `floor((W + 2P - K) / S) + 1`. For the stem,
`floor((128 + 2 - 3) / 2) + 1 = 64`. A 3×3, stride-1, padding-1 convolution
preserves width. The same calculation applies to height.

The first convolution has `3×24×3×3 = 648` weights; its bias is omitted because
batch normalization follows. Batch normalization adds 24 learned scales and
24 offsets, plus non-trainable running statistics. The final linear layer has
`192×177 + 177 = 34,161` parameters. Flattening all 8×8 spatial features into
the output layer would require 2,175,153 classifier parameters instead. Global
pooling controls model size but discards explicit spatial location.

## Input and output contract

One photograph receives one inherited plant-condition label. The task is
single-label classification, even if a real photo contains several plants or
symptoms. Detection, segmentation and a reliable multi-disease diagnosis are
outside scope. The 177 classes include species–condition combinations, healthy
tissue, pests, generic “diseased” categories and non-leaf plant parts.

The shared preparation selects frame zero, corrects camera orientation, converts
to RGB, resizes using bilinear interpolation while preserving aspect ratio and
adds gray padding. Pixels are divided by 255 and standardized by the saved
training-only per-channel mean and standard deviation. The final bundle records
its actual resolution and statistics. Never substitute an ImageNet normalization
or different crop at deployment.

Training augmentation uses whole-image reflections and quarter turns. These
preserve symptoms without cropping away a lesion or changing hue. Their realism
is limited for upright field scenes, which is a stated tradeoff. Evaluation and
inference are deterministic and use no augmentation. Augmented views do not
increase the count of independent plants.

The model returns logits, which are unrestricted scores. Cross-entropy accepts
logits and an integer class index. At inference, softmax converts the 177 logits
into positive scores summing to one. Apply it once. Argmax selects the class;
the ordered class list must match the saved output weights exactly.

## Why the splits and metrics matter

Training changes weights. Validation chooses methods, checkpoints and any
decision rules. Testing happens after all of these are frozen. Related views of
one leaf can share lighting, background and lesion shape; distributing them
across splits can reward memorization. Known groups remain together, but missing
specimen metadata prevents a guarantee of biological independence.

Accuracy measures correct images divided by total images. Precision asks what
fraction of predictions for a class are correct; recall asks what fraction of
that class's true images are found. F1 combines the two through their harmonic
mean. Macro-F1 gives each of the 177 classes equal weight. Frequent classes can
dominate accuracy while rare labels remain unrecognized. Confusion-matrix rows
are actual labels; columns are predicted labels.

Class weighting changes how much an example's loss contributes. The configured
weights depend only on training counts, are clipped to control extreme values,
and have an empirical training mean of one. Weighted training loss and unweighted
validation loss have different meanings. Compare the same validation metric
across methods and interpret weighted training loss within its own run.

The simple rule chooses the most common training class, never a class selected
from test frequencies. On validation it gets 1,462/16,836 correct, accuracy
8.684% and macro-F1 0.000903. Its test result is computed only after freezing the
final model. Earlier pilots and Create ML do not have the same comparison protocol.

## Understanding the training machinery

The forward pass creates logits. Loss compares them with labels. `zero_grad`
clears old gradients, `backward` calculates new gradients and `step` changes the
weights. Gradient clipping limits an update's gradient norm. AdamW adapts update
scales and applies weight decay. Dropout randomly removes features during
training to discourage dependence on particular activations.

Training mode enables dropout and updates batch-normalization statistics.
Evaluation mode disables dropout and uses stored normalization statistics;
`inference_mode` additionally disables gradient tracking. Validation accuracy can
exceed training accuracy because training has augmentation/dropout and measures
predictions throughout the epoch, while validation uses the final epoch weights.

The pipeline saves the best inference checkpoint immediately and atomically.
The resumable checkpoint additionally saves optimizer and scheduler states,
epoch/batch position, deterministic shuffle plan inputs, partial metric totals,
early-stop state and Python/NumPy/PyTorch/MPS random states. Six tests verify
contracts, metrics and resume, including real CPU and MPS interrupted runs.
CPU resumed model tensors match exactly; MPS checks use a small numeric tolerance.

The learning-rate scheduler reduces the step size after validation macro-F1
stalls. Early stopping gives reduced learning rates a chance before stopping.
Checkpoint selection uses macro-F1, then validation loss within the declared
tie tolerance; it never selects by test accuracy.

## Questions to practice

1. Why does a 177-label model need 177 logits rather than one “disease” score?
2. Which parts of the network did this project implement, and which calculations
   are supplied by PyTorch? Why is this still random-initialization training?
3. How do convolution, channel growth and downsampling change the features and
   tensor sizes? What would a larger resolution cost?
4. Why choose macro-F1? How can a model have high accuracy and poor rare-class
   performance? What does one correct image out of one prove?
5. Why can a file ID not identify a physical leaf? Which grouping evidence is
   documented, which is inferred, and what remains unknown?
6. What is held constant in each controlled comparison? Which experiments were
   actually completed, and what did their validation evidence show?
7. Why does the selected checkpoint sometimes precede the last epoch? What
   additional state is necessary to resume training rather than just predict?
8. Why is Create ML's score on the new holdout not a fair held-out comparison?
9. What exactly happens to a JPEG between upload and the network? How was website
   inference compared with standalone inference?
10. Why is a 99% softmax score not proof of correctness? What happens with an
    unfamiliar plant, non-plant object, multiple diseases or a different camera?
11. Which single next change follows from the observed validation evidence?
    Which observations after final testing must be left as future work?
12. Which work was student/partner work and which was assisted implementation?

## References

[1] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” in *Proc. CVPR*, 2016, pp. 770–778, doi:
[10.1109/CVPR.2016.90](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).
