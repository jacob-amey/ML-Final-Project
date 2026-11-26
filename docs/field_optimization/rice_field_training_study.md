# Source-separated rice field training study

The frozen model chose the wrong crop on only 15 of 968 master rice validation rows and 57 of 60 RiceLeafDiseaseBD field photos. Restricting the latter to rice produced fifteen correct labels, including zero healthy cases. These findings motivated a comparison of new field photographs and repeat draws at matched label exposure.

## Candidate sources and intake decision

- Training additions use [Munna RiceLeafDiseaseBD v3](https://data.mendeley.com/datasets/86s4jzj2m4/3), CC BY 4.0. The 420-candidate expansion retrieves only unmarked originals, excludes the 60 previous diagnostic file locators and restricts intake to healthy, brown spot and sheath blight. The filename-day cap is 22 per condition. After screening and the size cap, 125 photos per condition remain across 8 brown-spot and 10 healthy and sheath-blight filename-day groups. The source's blast naming has no role in this trial.
- [Hasan's field collection](https://data.mendeley.com/datasets/hx6f852hw4/1), CC BY 4.0, contributes 138 screened photos across the three exact labels. Its publisher test folder serves only as an external development source for this study. The healthy views span two observed days and both disease conditions span four days. All three conditions appear on October 6, 2023. Some observations share camera timestamps. Hash screening retained one view per matching same-label group and excluded label conflicts. The intended role here is held by the entire source.
- [BanglaRiceLeaf v2.1](https://doi.org/10.7910/DVN/XAOBYW) provides 54 healthy and 65 sheath-blight photos after screening and sample selection. Its dataset record declares CC0, and the [paper](https://doi.org/10.1016/j.dib.2026.113050) describes repeated views in BRRI fields. All 918 original archive entries for these two conditions lack EXIF dates. To limit sequence concentration, the intake uses at most one retained view per five adjacent archive numbers and per condition. These bins are proxies for file order, with no documented specimen identity. The source contains neither a scored brown-spot label nor a role in training.
- The [Rimi collection](https://data.mendeley.com/datasets/kx9rx8p2mz/1) was downloaded and checked against its published archive hash. All 252 healthy files had the same date and a Realme C21 camera tag. Every brown-spot filename matched a web-style pattern and lacked capture date and camera fields. The release archive included no CSV and had five more bacterial-leaf-blight image entries than the catalog. The naming pattern alone does not establish provenance. These findings leave class and acquisition conditions confounded. The Rimi archive remains outside the experiment.
- The dated healthy images in the Monirampur collection remain unscored and unassigned. They all come from one known day, which is separate from the brown-spot days.

The release screened 1,488 candidate photos against the frozen master, PlantDoc and previously registered field sources using raw bytes, decoded-pixel D4 identity and a visual hash radius of four bits. It also checked across the three new study sources and grouped likely repeated views within sources. The screen found 4 cross-label candidates in Hasan and 289 other repeated views across the collected sources. These hashes miss some transformations. Contact sheets show field framing and green foliage in both validation sources, with source-specific framing and some likely repeat views. Treat exact file counts and source-group claims separately from independent specimens.

## Controlled training and reproducibility

Both 160-pixel residual CNNs start with randomly initialized weights and train for twelve epochs on all 59,008 master training images plus 375 aligned additions. The control repeats seeded original images with replacement, matched label by label and slot by slot to the actual added field photos. The two arms share the optimizer, schedule, class weights, train-only normalization and batch order. Full-label master validation macro-F1 selects each arm's checkpoint. Ties retain the existing training rule. Compare eight-view predictions using all 177 outputs. In each source, average field F1 over its represented reference labels and count wrong-crop errors separately. The preregistered advancement rule considers both source cohorts, with master validation guards.

- Restore and verify the archive downloads against the [2023 source pins](../../data/manifests/leafcheck_rice_2023_sources.json), [BanglaRice pins](../../data/manifests/leafcheck_banglarice_v2_1_sources.json) and [Munna pins](../../data/manifests/leafcheck_rice_sources_v1.json).
- Place the two pinned BanglaRice RAR files under `data/raw/leafcheck_field_v13/rice/`, then extract each to a fresh folder named for its archive stem. Restore the pinned Munna workbook and perform the original acquisition or use its manifest, then run its expanded acquisition. The assigned source selection never touches test predictions.
- Run the following commands from the repository root. Choose fresh run directories when repeating a completed study. For an interrupted arm use the same `--run` directory with `--resume`, after confirming the earlier writer has stopped.

```sh
.venv/bin/python scripts/inspect_rice_field_sources.py --expanded
.venv/bin/python scripts/acquire_hasan_rice.py
.venv/bin/python scripts/prepare_rice_field_study.py --output data/processed/leafcheck_rice_field_study_v1
.venv/bin/python -m leafcheck.train --config configs/leafcheck_rice_field_matched_repeat_seed434.json --run runs/field_v13/reproduction/matched_repeat
.venv/bin/python -m leafcheck.train --config configs/leafcheck_rice_field_new_field_seed434.json --run runs/field_v13/reproduction/new_field
.venv/bin/python -m leafcheck.evaluate_rice_field_study --run runs/field_v13/reproduction --output runs/field_v13/reproduction/comparison
```

The rebuilt training and validation manifests must match the digests in the [registered plan](../../configs/leafcheck_rice_field_study_v1.json). The file-level source and inspection details are recorded in the [release audit](rice_field_release_audit.json).

To recover the registered run in place, use the sequential runner after the older launcher has stopped:

```sh
caffeinate -i .venv/bin/python scripts/run_rice_field_study.py --run runs/field_v13/rice-study
```

The runner refuses a second writer, checks the saved arm configuration, resumes incomplete arms and skips completed arms with reload evidence. It executes the registered validation comparison after both checkpoints finish. A live-writer check was exercised against the original sequence. Logs and checkpoints remain in the existing study directory.

## Deployed reference on the registered cohorts

The promoted 224-pixel model was replayed on CPU before either study arm was compared. The original 177 outputs were eligible. The reference scored 33/119 on BanglaRiceLeaf, with represented-label macro-F1 0.336735 and 48 wrong-crop predictions. It scored 3/138 on Hasan, with represented-label macro-F1 0.035088 and 105 wrong-crop predictions. All 36 correct results were sheath blight. The reference classified zero of 88 healthy and zero of 52 brown-spot photos correctly. Equal-day accuracy within Hasan was 0.01553. The source differences support tracking both crop and condition failures and keeping the two cohorts distinct.

[Reference metrics and hashes](rice_field_deployed_reference.json) · [Image-level replay](rice_field_deployed_reference.csv).
