# Six-minute LeafCheck rehearsal

Practice target for Jacob Amey and Demetre. Adjust to the instructor’s confirmed
time limit. Open the slides and stage the installed demo in advance. Keep one
authorized JPEG or PNG ready for file selection. The two included fixtures
work offline if a personal photo is unavailable.

| Elapsed | Speaker | What to show |
| --- | --- | --- |
| 0:00–1:00 | Jacob | Define the one-photo task and inherited labels. Show the 177-label grouped split and why known related images stay together. |
| 1:00–2:00 | Jacob | Show the architecture and explain random initialization, training-only statistics, and the paired choice of 224 pixels and eight-view averaging. |
| 2:00–2:45 | Jacob | Compare the three methods on 8,425 images. Name the original 0.85 target and measured 0.616940 macro-F1. Give the supplementary results separately. |
| 2:45–4:30 | Demetre | Launch the staged app, choose a fresh file, check its preview and run the comparison. Show the saved healthy tomato success and tomato late blight mistake. Point to the score explanation. |
| 4:30–5:15 | Both | Explain the field-photo and rare-class limitations, name the next data improvement and disclose roles and Codex assistance. Claim only work each person can substantiate. |
| 5:15–6:00 | Both | Take questions. |

If the fresh photo is a fixture, identify its source and use it solely as a
functional demonstration. A result from one photo is not a field benchmark.
The saved mistake is labeled tomato late blight in the dataset. The CNN chose
potato late blight with score 98.2%. The observed result gives no causal
explanation for the model’s choice.

## Questions to practice

- **How did you limit leakage?** Use a fixed split that keeps recorded related
  views together. Screen for exact and transformed duplicates and keep known
  previously used images in training. Fit RGB statistics only on the training
  split. Unknown specimen IDs limit the grouping evidence.
- **Why report macro-F1?** F1 balances precision and recall per class. Averaging
  equally across 177 labels gives rare labels the same weight as common labels.
  Accuracy weights each image equally. The primary test includes 69 classes with
  fewer than ten photos, so class-level estimates are uncertain.
- **Why 224 pixels?** The 224-pixel pair improved on 160 by 0.019429 mean
  validation macro-F1. The 256 pair gained only 0.003166 over 224, below the
  registered +0.005 threshold. Eight-view averaging gained 0.007758 at 224
  and passed the 500 ms CPU p95 validation guard.
- **What does the simple rule establish?** It predicts grape black rot, the
  largest training class, regardless of photo. The held-out comparison also
  includes a full-data Compact177 CNN using the same test membership.
- **What is a 98.2% model score?** It is an uncalibrated score used to rank the
  supported inherited labels. The saved mistake illustrates that a high score
  can accompany an incorrect prediction.

Local technical rehearsal was verified. Student rehearsal and attendance
await the teammates’ confirmation.
