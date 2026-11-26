# Tanzanian common bean intake

The [publisher archive](https://zenodo.org/records/15685315) is CC BY 4.0. The [data paper](https://doi.org/10.1016/j.dib.2026.112909) describes smartphone field collections in Northern Tanzania with agricultural review. The selected sample has 24 published JPEG members, eight each from `healthy1.zip`, `rust1.zip`, and `anthra.zip`. The acquisition script ranks the archived filenames by SHA-256, fetches only the required byte ranges, validates the ZIP CRC and decodes the original JPEGs. The intake manifest pins all 24 names, byte digests and available camera metadata. Publisher archive MD5 values are reference identifiers because the full archives were not fetched.

The frozen model can return halo blight, mosaic virus and rust for beans. This source supplies one overlapping condition, rust, and two absent conditions, healthy and anthracnose. This preflight checks how the application responds to those absent conditions. For each condition report the count that receives a suggestion, the count assigned to bean, correct supported labels for rust, and score parity across all 177 labels. Treat the counts as diagnostic evidence from one source. No threshold fitting, test reporting, label change, or training admission is part of the preflight.

A contact sheet reviewed before inference showed mostly leaf photos with varying framing, including close-ups on the underside. `healthy1/1712049935863.jpg` had a strap-like leaf and was flagged for uncertain crop identity. The sampled healthy photos are dated March–April 2024 while the disease photos are dated January–April 2023. Several anthracnose images from the same date have similar vein patterns. All 24 report EXIF `SM-A032F`. Group by recorded date for description. Independent plant counts, the camera instance and exact site remain unknown. Treat the whole collection as one source group when planning future evaluation. Any future dataset development must control how year, framing and other acquisition conditions align with the class.

Recreate and screen from the repository root with:

```sh
.venv/bin/python scripts/inspect_tanzania_bean.py
.venv/bin/python scripts/inspect_tanzania_bean.py --screen runs/field_v12/bean-screen.json
```
