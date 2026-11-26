# Tanzanian common bean intake

The [publisher archive](https://zenodo.org/records/15685315) is CC BY 4.0. The [data paper](https://doi.org/10.1016/j.dib.2026.112909) describes smartphone field collections in Northern Tanzania with agricultural review. The selected sample has 24 published JPEG members, eight each from `healthy1.zip`, `rust1.zip`, and `anthra.zip`. The acquisition script ranks the archived filenames by SHA-256, fetches only the required byte ranges, validates the ZIP CRC and decodes the original JPEGs. The intake manifest pins all 24 names, byte digests and available camera metadata. Publisher archive MD5 values are reference identifiers because the full archives were not fetched.

The frozen model can return halo blight, mosaic virus and rust for beans. This source supplies one overlapping condition, rust, and two absent conditions, healthy and anthracnose. This preflight checks how the application responds to those absent conditions. For each condition report the count that receives a suggestion, the count assigned to bean, correct supported labels for rust, and score parity across all 177 labels. Treat the counts as diagnostic evidence from one source. No threshold fitting, test reporting, label change, or training admission is part of the preflight.

A contact sheet reviewed before inference showed mostly leaf photos with varying framing, including close-ups on the underside. `healthy1/1712049935863.jpg` had a strap-like leaf and was flagged for uncertain crop identity. The sampled healthy photos are dated March–April 2024 while the disease photos are dated January–April 2023. Several anthracnose images from the same date have similar vein patterns. All 24 report EXIF `SM-A032F`. Group by recorded date for description. Independent plant counts, the camera instance and exact site remain unknown. Treat the whole collection as one source group when planning future evaluation. Any future dataset development must control how year, framing and other acquisition conditions align with the class.

Recreate and screen from the repository root with:

```sh
.venv/bin/python scripts/inspect_tanzania_bean.py
.venv/bin/python scripts/inspect_tanzania_bean.py --screen runs/field_v12/bean-screen.json
```

## Upload results

The first screen flagged one suspected non-bean leaf. The other 23 photos passed byte, decoded-pixel and visual-hash screening against 90,574 master records and 183 PlantDoc photos. The automated within-sample check flagged zero additional files, which leaves the observed same-date and specimen uncertainty in place.

| Source label | Screened photos | Suggested labels | Top label was bean | Correct supported label |
| --- | ---: | ---: | ---: | ---: |
| Healthy | 7 | 1 | 0 | Outside label map |
| Rust | 8 | 3 | 2 | 0 |
| Anthracnose | 8 | 5 | 5 | Outside label map |

Original JPEGs passed through the freshly extracted app. Every final-model label and all 177 scores matched standalone inference exactly. One anthracnose photo received bean rust with a model score of 0.9837. This confirms that the current score rule can still offer an unrelated condition on a familiar crop. Treat these as diagnostic counts with inherited labels from one source.

In the frozen master, bean rust has 76 training images in 76 recorded groups. It was predicted correctly on 5 of 22 master validation images and on 0 of the 8 Tanzanian rust photos. Those groups are recorded identities and do not establish independent field specimens. Rust is the only eligible training label from this new source under the current 177-label scope. The new browser lookup shows the exact three bean labels without changing the inference path.

[Upload evidence](tanzania_bean_uploads.json) · [Browser verification](condition_coverage_browser.json) · [Demo archive verification](condition_coverage_archive_verification.json).
