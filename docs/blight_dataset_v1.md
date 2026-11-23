# LeafCheck blight training and validation supplement

The local release lives at `data/processed/leafcheck_blight_v1/`. It targets
tomato late blight, tomato early blight, healthy tomato, and potato late blight.
Keep the original 177-class training data as the base for a broader model.
The supplement preserves the original 177-class ID mapping. Its four folders
per split are convenient for browsing. Load labels from the manifest or
explicitly map folder names to the saved IDs when using a folder-based loader.

| Label | Train | Validation |
| --- | ---: | ---: |
| Tomato late blight | 436 | 17 |
| Tomato early blight | 406 | 57 |
| Healthy tomato | 285 | 186 |
| Potato late blight | 19 | 20 |
| **Total** | **1,146** | **280** |

Of the 1,636 candidates, 1,426 passed screening. The validation set has 233
photos from held-back collection dates, 39 from a validation-only leaf
collection, and eight individually attributed field photos. Five of the field
photos are tomato late blight. All 17 validation images for that label come
from sources reserved for validation and span four recorded source or
photographer groups.

## Sources and roles

| Source | Role and selection | Terms |
| --- | --- | --- |
| [Sheikh et al., Tomato Leaf and Fruit Disease Image Dataset](https://data.mendeley.com/datasets/swtt5wy2pr/1) | Publisher-resized leaf images linked to the original metadata by class and filename. Entire capture dates assigned together. The earlier healthy-leaf date and later early-blight date supply same-source validation. All of the tomato late-blight candidates were captured on one date and assigned to training. | CC BY 4.0 |
| [Mduma and Mtali, Irish Potato Leaf Image Dataset](https://zenodo.org/records/17553016) | Training candidates from the published `LATEBLT_6.zip`. Keep the collection together and collapse repeated capture timestamps and visually similar views. The broader source release has other archives. | CC BY 4.0 |
| [Banu and Hossain, Solanaceae Family Leaf Disease Image Dataset](https://data.mendeley.com/datasets/sd5m3mgvvx/1) | A validation-only selection from the linked original-image folders. Accept plain `IMG_<number>.jpg` files and exclude files named as transforms, even when they appear in an original folder. Keep the selected collection together. Exact folders and file links are recorded in the manifest. | CC BY 4.0 |
| [Bugwood field images](https://www.ipmimages.org/) | Validation-only garden and field views. Each image has a verified crop-condition catalog record, photographer credit, and individual license. Keep each photographer's images in the validation split. Some published renditions show a small source number. | CC BY 3.0 or CC BY-NC 3.0, per image |

The release keeps the publisher's delivered image bytes. Labels come from the
source annotations and lack independent diagnostic confirmation by LeafCheck.
Many of the Sheikh and Banu photos show individual leaves on a plain
background. The Bugwood photos add a small field check. Acquisition dates,
camera characteristics, and photo backgrounds vary with class and source.
Treat the grouped validation images as correlated when interpreting results.

## Screening and rebuild

Every candidate was decoded, assigned a stable content ID, and checked using
raw SHA-256, decoded RGB SHA-256 invariant to rotations and reflections, and
perceptual hashing. The overlap search used the recorded master audit of
90,574 images, covering every frozen split, and all 183 registered PlantDoc
supplement photos. It also screened candidates against one another. Clusters
with multiple split assignments or competing labels were quarantined.
Perceptual matches are screening candidates. The method can miss heavy crops,
edits and additional photos of the same specimen.

The screen set aside 14 possible matches to the existing datasets, 17
candidates from clusters spanning train and validation, 92 candidates from
clusters with conflicting labels, and 87 repeated or highly similar views.
The visual review covered a fixed sample of 24 Sheikh photos and 16 potato
photos from the training sources, plus the 61 Banu candidates and eight
Bugwood candidates. Photo sources with unclear underlying rights, apparent
augmentations, or unverified class mappings were left outside the release.

`manifest.csv` contains training or validation membership, class ID, capture
group, source archive member or image link, attribution, license, file hash, and
reasons for exclusions. `split_summary.csv` gives exact label counts.
`release.json` pins the manifest, mapping, counts and source records.
Use the original publisher links and credits when sharing this educational
dataset. Several field images have noncommercial terms.

From the repository root with the existing research environment and frozen
release available:

```sh
.venv/bin/python scripts/prepare_blight_dataset.py fetch
.venv/bin/python scripts/prepare_blight_dataset.py build
.venv/bin/python scripts/prepare_blight_dataset.py verify
```

`fetch` verifies each pinned original and downloads missing files. `build`
refuses to overwrite an existing release. To reproduce and compare fingerprints,
pass a fresh path with `build --output /absolute/path/to/new-copy`. Image bytes
remain local. Git tracks the preparation code and source pins.

A fresh build reproduced fingerprint
`c2147459192fcc2d49d7787162960786353302220274719d512dedb76ea7cb78`.
The 224-pixel LeafCheck preprocessor produced finite RGB tensors from one
photo per class and split, with the expected original class IDs.

## Use in a follow-on study

For another 177-class run, add the supplement's `train` images to the original
training split in a versioned training configuration. Keep the original
validation split and measure the new validation supplement as an additional
slice. Report the field-photo results with counts and source information.
Use a new independent test protocol for any new final performance claim.
This release leaves the frozen dataset, model, evaluation, and demo in place.
