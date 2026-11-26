# Colombian coffee field confirmation

The [CoffeeLeaf-CO v2 release](https://zenodo.org/records/22756170) documents
a Colombian field collection and a separately sourced Brazilian subset.
Only the publisher-declared Colombian collection enters the prospective
LeafCheck confirmation. The records and archive use CC BY 4.0. The raw
originals stay local.

The published CSV retains 454 removed RoCoLe rows and lists 802 of the
1,705 Colombian images under different split paths than the current ZIP.
The intake used unique archive filenames, verified the matched content
hashes and read the corresponding annotations in the actual archive
directories. The publisher split column was not used for LeafCheck
membership.

The catalog contains 158 eligible single-label rust annotations and
1,113 healthy backgrounds. The archive sidecars for the checked rust
images contain only class ID 0. Healthy samples use the publisher's
healthy-background label, an empty annotation and its healthy source
prefix. That is a visual proxy for healthy. The deterministic filename
hash selected 60 originals per class. A contact sheet of 24 showed outdoor
framing, with a few leaves held or placed against other material. This
sample had no recorded EXIF capture days.

The screen excluded six possible visual matches to master records. The
remaining 60 healthy and 54 rust photos form one source group with
fingerprint `b8b4288addd3421e52ab1e66edcfab01a601d96f7fd36382c669ac1213f11eef`.
Neither fitting nor baseline inference used the cohort. Preserve it for a
prospectively declared candidate. Its two represented conditions and
single site limit what it can say about the full 177-label task.

[Publisher archive record](../../data/manifests/leafcheck_decafia_v2_source.json) ·
[Exact source and labels](../../data/manifests/leafcheck_decafia_v2_originals.csv) ·
[Screening results](../../data/manifests/leafcheck_decafia_v2_manifest.csv) ·
[Release seal](../../data/manifests/leafcheck_decafia_v2_release.json).

A separate reconstruction reproduced the manifest, source records and
release seal byte for byte. Future candidate intake must check against
this fingerprint before fitting or scoring. The two labels remain in
the original 177-label output space.
