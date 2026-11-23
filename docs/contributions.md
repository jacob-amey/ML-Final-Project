# Team contributions

Record actual work and decisions as the project progresses. Everyone should
understand the data, models, evaluation, and demonstration.

## Responsibilities

| Member | Primary responsibilities |
| --- | --- |
| Model/dataset team member (user) | Dataset, custom CNN, training, evaluation and integration handoff |
| Project partner (name not supplied) | Frontend/application/interface, as described by the user |
| Codex assistance | Dataset preparation and final execution/verification explicitly delegated in this conversation; work is identified in the log |

## Activity log

| Date | Member | Work completed | Commit, issue, or document |
| --- | --- | --- | --- |
| 2026-11-15 | Codex-assisted work | Audited release, grouped splits, exclusions and evidence | `9e730c7`; `docs/final_dataset_audit.md` |
| 2026-11-15 | Codex-assisted work | Custom training/inference pipeline, training-only statistics, diagnostics and exact CPU resume tests | `8ac19f2`; `docs/final_model_work_log.md` |
| 2026-11-15 | Pre-existing local demo work, authorship not independently established | Existing migration, bundling and UI preserved before final-model changes; nine integration tests passed | Nested demo `22e454f`; preflight backups identify original changes |
| 2026-11-16 | Codex-assisted work | Completed full-data architecture, class-weight, duration and paired-seed resolution experiments; preserved all177 labels and the sealed holdouts | `docs/experiments.csv`; `docs/resolution_decision.json` |
| 2026-11-16 | Codex-assisted work | Implemented final-model packaging, evaluation and website integration tooling; final artifact tests remain pending until selection | Source and `docs/final_model_work_log.md` |
| 2026-11-22 | Codex-assisted work | Recovered the native-resolution repeat, completed paired selection, froze and evaluated the selected model and fair references | `3443b41`, `1e65c57`, `docs/final_model_report.md` |
| 2026-11-22 | Codex-assisted work | Installed the frozen model, verified standalone/API/browser agreement and prepared the report, defense deck and archived review notebook | `b0b8720`, `docs/demo_verification.json`, `docs/test_results.json` |

Do not infer attendance, independent student implementation or partner authorship
from commits. The student should review and understand assisted work and follow
any additional course AI-assistance policy. Add actual student/partner activity
rather than assigning completed-work credit from responsibility alone.

## Meeting decisions

| Date | Attendees | Decision / next action | Owner |
| --- | --- | --- | --- |
