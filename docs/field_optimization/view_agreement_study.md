# Can eight-view agreement flag unreliable uploads?

[The registered study](../../configs/leafcheck_view_agreement_v1.json) compares the deployed top score with disagreement among the existing rotations and reflections. It computes Jensen–Shannon disagreement from the eight 177-class probability distributions, with low disagreement treated as stronger agreement. A secondary diagnostic counts how many views choose the label selected by their average. For each cohort a replay verifies the averaged probabilities against the shared inference implementation. The use of prediction agreement was motivated by [the test-time aggregation study](https://openaccess.thecvf.com/content/ICCV2021/html/Shanmugam_Better_Aggregation_in_Test-Time_Augmentation_ICCV_2021_paper.html).

| Ranking task | Deployed score AUROC | View disagreement AUROC | View top-label agreement AUROC |
| --- | ---: | ---: | ---: |
| Correct vs. incorrect predictions, pooled supported field development | 0.5142 | 0.5953 | 0.5339 |
| Supported field vs. unrelated objects, mean across sources | 0.6140 | 0.5633 | 0.5611 |
| Supported field vs. unsupported plant photos, mean across sources | 0.5043 | 0.5006 | 0.4211 |

The supported data contain 35 correct top labels across 680 photos from four sources. The unsupported data contain 22 unrelated objects and 23 unsupported plant photos. Sheikh supplies zero correct predictions, so its source-specific correctness AUROCs are undefined. Disagreement improved correctness ranking in the other three sources but weakened unrelated-object separation and left unsupported-plant discrimination near chance. Its primary signal failed the registered criteria. No calibration or threshold was fitted and the app continues to use the verified deployed rule.

This was a ranking experiment on existing development data. The aggregate AUROCs do not establish correctness probabilities or reliable rejection of unfamiliar photos. The exact cohort fingerprints, source measurements and decision are in [the result](view_agreement_result.json). Reproduce with `.venv/bin/python -m leafcheck.view_agreement_study --output runs/field_v13/view-agreement-reproduction`.
