# Crop loss and unsupported upload studies

The deployed 177-label CNN keeps its 224-pixel checkpoint, eight-view probability average and decision rule. The matched crop-loss run missed its advancement criteria. An upload study showed that the current rule can suggest a supported label for an unrelated photo. The demo now lists its 43 covered crops and explains what a high score means outside those crops.

## Crop loss screening

The candidate adds 0.5 times a crop-level cross-entropy to the usual weighted condition loss. It sums the probabilities of all conditions of the correct crop before taking the log. This rewards crop recognition while retaining the original 177 joint outputs. See the [registered protocol](../../configs/leafcheck_crop_loss_pilot_v1.json) and the related [hierarchical classification paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Bertinetto_Making_Better_Mistakes_Leveraging_Class_Hierarchies_With_Deep_Networks_CVPR_2020_paper.html).

Both seed-434 runs started with randomly initialized weights. Each epoch used the same 59,008 master training images, 128-pixel preprocessing, batch schedule and augmentation policy. Six epochs took 265 seconds for the control and 269 seconds for the candidate on the M2 Max. The best checkpoints reloaded successfully. The field metric is the equal-weight mean of each source's macro-F1 over its represented labels, with all 177 predictions eligible. Mean wrong-crop rate also weights the three sources equally.

| Model | Master validation macro-F1 | Master accuracy | Field macro-F1 | Field correct |
| --- | ---: | ---: | ---: | ---: |
| Existing deployed 224-pixel model | 0.649221 | 15,111/16,836 | 0.050016 | 14/586 |
| Matched six-epoch control | 0.369417 | 12,380/16,836 | 0.000000 | 0/586 |
| Matched six-epoch crop loss | 0.379271 | 12,362/16,836 | 0.017231 | 6/586 |

The candidate improved field macro-F1 by 0.017231 and reduced wrong-crop rate by 2.64 points. The registered minimums were 0.020 and 3 points. This ended the pilot before a repeat seed. The 224-pixel incumbent has a different training budget and serves as a deployment reference. This comparison cannot isolate a causal effect between it and either six-epoch arm. The reserved evaluations were outside the study.

Reproduce the six-epoch arms with the [128-pixel recipe](../../configs/residual177_128_color.json), seed 434, six maximum and minimum epochs, and crop loss weights 0 and 0.5 in separate run directories. Run `.venv/bin/python -m leafcheck.evaluate_crop_loss_pilot --run runs/field_v8 --output runs/field_v8/crop_loss_replay.json` once both completed runs are present. The [result](crop_loss_pilot.json) identifies the checkpoint and release fingerprints and contains the terminal records and per-source errors.

## Supported and unsupported uploads

The 45 admitted development photos comprise 22 bicycles, dogs or typewriters and 23 cactus, orchid or spruce photos. Three of the 48 photographed candidates were excluded during visual intake. The positives are 586 labeled images from the grouped Sheikh, Mduma and Pakistan development cohorts. See the [Open Images intake and restoration notes](../../data/README.md#unsupported-upload-development-photos).

The deployed threshold suggested a label for 6/22 nonplant photos and 7/23 unsupported plant photos. It also suggested a label for 186/586 supported field photos, of which one was correct on the full crop-condition target. This curated challenge measures specific failure modes and has too few categories to estimate a general upload error rate.

Margin, entropy, total crop probability and fixed raw-logit scores failed to support a threshold replacement. The [fixed-signal results](unsupported_energy.json) include the several baselines, the acceptance counts and paired AUROCs. The current threshold was examined in development without refitting it.

A [registered frozen-feature probe](../../configs/leafcheck_upload_support_probe_v1.json) went further by fitting a small regularized linear model on pooled CNN features. Each of three folds held out one field source and one category of each unsupported type. Macro AUROC rose from 0.570 to 0.770 for nonplants and from 0.461 to 0.622 for unsupported plants. The latter missed 0.65. On the held Sheikh/cactus pair the probe fell from 0.622 to 0.306. That variability and the missed gate leave the deployment rule unchanged. See the [fold results](upload_support_probe.json) and [replay scores](upload_support_probe_predictions.csv). No threshold or calibration was fit.

## Upload and delivery verification

The [original-byte API replay](unsupported_upload_replay.json) matched all 177 standalone scores exactly on the typewriter and spruce. The 93.8% raspberry fire blight score and 92.5% corn smut score demonstrate that a high score can appear on an unsupported subject. A [fresh-extraction browser check](field_guidance_browser.json) reproduced the typewriter result, returned healthy tomato on the existing success fixture, and displayed the 43-crop coverage list and revised score explanation.

The [fresh-extraction archive check](field_guidance_archive_verification.json) verified four ready models and 90 manifest entries on Apple silicon. The healthy tomato and tomato late blight fixtures reproduced their recorded outcomes. Invalid bytes returned HTTP 400 and an unsupported format returned HTTP 415. Warm loopback HTTP p95 was 53 ms over 12 measured requests. This measures the upload and four-model response on the development Mac.

The separately packaged demo is `artifacts/LeafCheck-field-guidance-demo.zip`. The editable academic update is `artifacts/submission/LeafCheck_Report_Field_Study_Rev4.docx`. The original course archive remains the record of the frozen submission.
