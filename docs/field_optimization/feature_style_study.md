# Mixing early CNN feature statistics

The [registered experiment](../../configs/leafcheck_feature_style_v1.json)
tested whether variation in early feature channels could reduce sensitivity
to source appearance. The intervention corresponds to the published
[MixStyle approach](https://openreview.net/forum?id=6xHJ37MVxxp). It mixes
spatial means and deviations from random images within a training batch
after the first two residual blocks. The implementation preserves the
labels, base image augmentation, 177 outputs and inference graph. The
mixing RNG is derived from seed, epoch and batch and leaves the base
augmentation RNG alone.

Both random-initialization runs trained for eight epochs on 59,008
images per epoch. Their saved checkpoints reloaded at epoch eight. An
interruption after three updates verified the treatment's resume entry
point. A regression test also matched four uninterrupted CPU updates
against two updates followed by a two-update resume with forced mixing.

| Model | Master validation accuracy | Master macro-F1 | Four-source mean field macro-F1 | Field correct |
| --- | ---: | ---: | ---: | ---: |
| Deployed 224-pixel CNN | 15,111/16,836 | 0.649221 | 0.087027 | 35/680 |
| Matched 128-pixel control | 12,342/16,836 | 0.402295 | 0.028728 | 7/680 |
| Feature style mixing | 12,574/16,836 | 0.399008 | 0.030998 | 15/680 |

All field predictions use 177 eligible outputs. The primary screen
averages the four source macro-F1 scores equally, with each score
averaged over that source's represented labels. The counts weight each
photo equally. The deployed result uses a different resolution and
training budget and remains the operating reference.

Mduma and Pakistan gained 0.053963 and 0.041363 macro-F1. Sheikh stayed
at zero. Pretoria lost 0.086244 macro-F1 and went from six to one
correct. This exceeded the per-source loss limit of 0.020. The mean
gain of 0.002270 also missed +0.020. The master macro-F1 difference
was −0.003288. The result ended the trial at seed 434. It supports no
advance to a longer run for this recipe. The study's short schedule limits
conclusions about other ways of mixing features or optimizing longer.

[Exact scores, history and fingerprints](feature_style_study.json).
Held-out and reserved confirmation images were outside the comparison.
