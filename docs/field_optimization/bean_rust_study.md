# Matched bean rust study

The frozen 177-class CNN returned bean rust for 5 of 22 master validation photos and none of eight screened Tanzanian rust photos in the intake. The master has 76 rust training photos in 76 recorded groups. The new experiment asks whether exposure to diverse field photos helps more than repeating the same label at the same frequency.

## Training and validation membership

The [Makerere AI Lab release](https://github.com/AI-Lab-Makerere/ibean) describes field photography in Uganda and annotation by NaCRRI specialists. The publisher links the [pinned Hugging Face mirror](https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/27aa014ce09b193e1a6f58112d4a66e0eddb69c5). Its 2020 ZIPs matched the three published archive SHA-256 digests. The archives contain 1,295 JPEGs. A contact sheet of 30 publisher-training photos showed a mix of held leaves and natural foliage with variable framing. The files provide no EXIF capture time. Treat the training addition as one source group and avoid reconstructing farm or specimen membership from filenames.

Master, PlantDoc and prior field release screening found three possible visual matches. All were marked in the audit and excluded from selection. Two concern rust from the publisher's test archive, which is unused. One concerns bean angular leaf spot, a condition outside the current label map. All 348 eligible publisher-training rust JPEGs passed the automatic overlap screen. The training manifest records the exact archive member and raw SHA-256 for each. Healthy and angular leaf spot remain outside the frozen labels. Similarity screening has limited sensitivity for severe crops or changes in angle.

The development comparison uses 63 additional JPEGs from the [Tanzania publisher's rust archive](https://zenodo.org/records/15685315) under CC BY 4.0. It excludes the original eight rust photos that informed source selection and one possible visual overlap from the expansion. The master, Tanzania preflight and Makerere training originals were included in the overlap check. Most images are close views of visible spots. Contact sheets showed likely repeated capture views, particularly within the February 9 group, and a few leaves with less obvious symptoms. Date metadata provides ten coarse groups and the whole source is one group. The analysis reports pooled recall, per-day counts and an equal-day summary. Each file inherits the publisher label. Date groups and representative files help show concentration without certifying independence or diagnosis. This is a one-condition check of sensitivity with all 177 outputs eligible.

## Matched baseline and recovery

Both scratch residual CNNs have 724,713 parameters, 128 × 128 resolution, seed 434, and eight full epochs. They use the same original 59,008 rows plus 348 slots labeled bean rust. The control fills these slots with a deterministic sample, with replacement, of the 76 master training photos. The treatment uses the new Makerere bytes in those slots. Epoch order, class weighting, augmentation, total updates and the original training-only color statistics match between arms. This compares new image content with extra draws under the same class exposure.

Use original 177-class validation macro-F1 for checkpoint selection. Score both selected states with eight-view probability averaging on master validation and the registered Tanzania rust cohort. The control is the only matched comparator. The 224-pixel deployed model provides context under a different training budget. The [registered plan](../../configs/leafcheck_bean_rust_study.json) specifies what would justify a second seed. The course test, the Colombian coffee confirmation and all other reserved evaluations remain outside this comparison.

From the repository root, restore the raw archives from the pinned URLs and verify their digests against [the audit](makerere_beans_audit.json). Acquire the Tanzanian validation originals with the following command. The expanded cohort scope is fixed by the versioned manifest.

```sh
.venv/bin/python scripts/inspect_tanzania_bean.py --expand-rust
```

Train to new run directories when recreating the comparison. Use the original training and validation caches with the frozen release.

```sh
.venv/bin/python -m leafcheck.train --config configs/leafcheck_bean_rust_matched_repeat.json --run runs/field_v12/reproduction/matched_repeat
.venv/bin/python -m leafcheck.train --config configs/leafcheck_bean_rust_new_field.json --run runs/field_v12/reproduction/new_field
.venv/bin/python -m leafcheck.evaluate_bean_rust --run runs/field_v12/reproduction --output runs/field_v12/reproduction/comparison
```

For an interrupted arm use `--resume --run <run-directory>` with its saved configuration and checkpoint. Check the running process and its configured output path before starting a second job. The training code revalidates the manifest and original archive on resume. Full-data training starts from randomly initialized weights in both arms.

## Measured comparison

Both runs finished all eight epochs and 3,712 updates on MPS. Both selected epoch 8 and passed checkpoint reload. Eight-view inference gave the following results.

| Model | Master validation accuracy | Master validation macro-F1 | Tanzania rust correct | Tanzania wrong crop |
| --- | ---: | ---: | ---: | ---: |
| Deployed 224-pixel reference | 89.7541% | 0.649221 | 1/63 | 60/63 |
| 128-pixel repeat-draw control | 79.2825% | 0.436075 | 0/63 | 63/63 |
| 128-pixel Makerere photo addition | 76.6631% | 0.423678 | 0/63 | 62/63 |

The treatment lost 0.012397 master macro-F1 and 2.6194 accuracy points. Master rust recall fell from 5/22 to 2/22. The full source, equal-date and one-file-per-date summaries produced zero correct rust predictions for either pilot. The treated model labeled one field leaf as bean with the wrong condition. Both other bean labels together gained one correct master prediction. The original nonrust preflight is a descriptive replay. The deployed model labeled five anthracnose uploads as bean rust, the control labeled one, and the treatment labeled zero.

The treatment failed the full-label accuracy and macro-F1 limits and produced zero field recall gain. Skip a second seed under the registered rule. The two short pilots cannot establish whether a longer recipe or more independently sampled sources would help. The prevalence of incorrect crop predictions and photographed close-ups both support investigating camera distance and ordinary crop cues. The evidence does not identify a causal reason for the errors. Retain the verified frozen checkpoint in the app.

[Full metrics, hashes, timings and group counts](bean_rust_result.json).
