# Rice field source intake

The deployed CNN called 49 of 196 master validation leaf-blast photos healthy. The frozen classes include `rice__blast`, `rice__leaf_blast` and `rice__neck_blast`. Their original source folders differ. The first two both refer to blast in the available names, but those names do not define distinct biological conditions. Preserve the 177 labels. New sources that only say “Blast” can inform a disease-family diagnostic without forcing an unsupported one-to-one training mapping. Neck blast remains separate because these photos show leaves.

The [RiceLeafDiseaseBD v3 publisher](https://data.mendeley.com/datasets/86s4jzj2m4/3) supplies a verified 9,769-image ZIP catalog with separately named unmarked images, visual overlays and annotations. The archive directory suffix, metadata spreadsheet, annotation protocol and README are pinned locally. The v3 workbook has 9,769 populated filename mappings and nine repeated original-name entries above unique count. The intake selects only uniquely named unmarked originals with a usable capture-day hint. Several classes appear across multiple dates. Filename dates are grouping hints and cannot identify a distinct plant or plot.

The fixed 60-photo preflight has 15 each for healthy, brown spot, blast and sheath blight, with at most four per recorded filename day and condition. All 60 passed the raw-byte, decoded-pixel, perceptual and within-sample checks against the master and registered PlantDoc set. The preflight contact sheet showed leaves under varied outdoor lighting with much of the field visible around the main leaf. Lesions are often small. No overlay images enter the sample. The dataset supplies one source group. Each label is inherited from the publisher.

The separate [RiceLeafBD v1 archive](https://data.mendeley.com/datasets/sjkptpgv79/1) lists 3,150 JPEGs in the downloaded RAR versus 2,545 in its description. All archive JPEG names carry one of three dates in late October or early November 2024, with blast and tungro restricted to the same single date. There are repeated timestamp stems and filename suffixes within classes. Determining which files represent original specimens would require additional reconciliation. The v3 workbook's original basenames did not match v1 exactly, which is insufficient to rule out resized, renamed or cropped overlap. Leave v1 unassigned.

For the v3 diagnostic, compare the frozen automatic top label with the highest-scoring rice label when the crop is given. This replay uses the exact three supported condition labels and accepts either leaf-oriented blast label only for the blast family. Report correct/total by condition, wrong-crop counts, date-group sensitivity, and exact top labels including the two blast alternatives. This diagnostic can show which type of error is common. It provides no promotion or calibration evidence and has no effect on the deployment or course split.

[Intake and publisher hashes](../../data/manifests/leafcheck_rice_sources_v1.json) · [Fixed screened membership](../../data/manifests/leafcheck_rice_field_preflight.json).

## Frozen diagnostic results

| Publisher condition | Automatic correct | Automatic wrong crop | Correct given rice |
| --- | ---: | ---: | ---: |
| Healthy | 0/15 | 15/15 | 0/15 |
| Brown spot | 0/15 | 14/15 | 1/15 |
| Blast family | 1/15 | 14/15 | 5/15 |
| Sheath blight | 1/15 | 14/15 | 9/15 |
| Total | 2/60 | 57/60 | 15/60 |

The highest rice score improved leaf-family and sheath-blight predictions. Healthy recall remained zero and the majority of the brown-spot samples still became generic blast. All five correct blast-family conditional predictions used `rice__blast`; `rice__leaf_blast` received none. The original rice cases came from one publisher and can include related field specimens. This study supports separating crop errors from condition errors and collecting representative photos of the exact healthy and brown-spot classes. It does not endorse a new known-rice upload choice.

With the original bytes restored, run:

```sh
.venv/bin/python -m leafcheck.rice_field_diagnostic --output runs/field_v12/rice-diagnostic-repeat
```

[Metrics and captured dates](rice_field_diagnostic.json) · [Original top labels and scores](rice_field_predictions.csv).

The extracted comparison app replayed two original blast JPEGs. One matched
the blast family at a 68.6% model score. The other received sugarcane red rot
at 99.3%. Both produced suggestions and all 177 scores matched standalone
inference exactly. [Packaged upload replay](rice_upload_replay.json).

On the frozen master validation rows, the same checkpoint chose the wrong
crop on 15 of 968 rice photos and got the exact joint label right on 782.
On RiceLeafDiseaseBD, it chose wheat for 23 and sugarcane for 18 of the 60
photos. The field setting changes much more than class frequency. Framing,
background and capture conditions can vary by source, and this comparison
does not distinguish their individual effects.
