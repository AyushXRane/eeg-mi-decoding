# Experiment log

One row per run. Filled in when the run finishes, not afterwards from memory.
Machine: M-series Mac, 15 or 30 subjects as noted, imagined runs 4/8/12 unless stated.

| # | Date | Experiment | Config | Result | Conclusion |
|---|------|-----------|--------|--------|------------|
| 1 | 2026-09-07 | A1 T1/T2 verification | runs 3/7/11 + 4/8/12, S001–S005 | 160 Hz, T0 45 @4.1–4.2 s, T1 22–23 @4.10 s, T2 22–23 @4.10 s, 7–8 trials/class/run | Matches SPEC §3 exactly. **But annotation structure is identical in runs 5/9/13 (fists/feet)** — counts and durations give no way to tell the two paradigms apart. Only the run number does. |
| 2 | 2026-09-07 | A1b physiological label check | C3/C4 mu lateralisation, T1 vs T2, executed runs | see results/label_check.txt | Verifies the mapping by physiology instead of by trusting the docs. |
| 3 | 2026-09-07 | Data QC | 32 subjects attempted | 45 trials/subject (15 per run), balanced 22/23 | No subject in 1–32 failed QC beyond the four excluded by construction. |
| 4 | 2026-09-07 | EA implementation bug | logvar subject-ID probe, 8 subjects | 0.989 with EA (expected collapse) | **Bug, not result.** CAR makes the EA reference covariance singular; an absolute eigenvalue floor amplified the empty all-ones direction ~1e5× and that numerical noise was itself subject-identifying. Switched to pseudo-inverse square root (drop eigenvalues < 1e-6·max). Probe then went 0.989 → 0.014. |
| 5 | 2026-09-07 | B1 random 5-fold pooled | 15 subj, imagery | csp_lda 0.535 ± 0.042, tangent 0.643 ± 0.052 | The deliberately wrong protocol. |
| 6 | 2026-09-07 | B2 LOSO imagery→imagery | 15 subj, imagery | csp_lda 0.547 ± 0.093 (0.40–0.76), tangent **0.609 ± 0.085** (0.51–0.80) | The honest number. In the 0.55–0.65 band SPEC §2 predicted. |
| 7 | 2026-09-07 | B3 leakage gap (B1−B2) | 15 subj | csp_lda **−1.2 pp**, tangent **+3.4 pp** | **Prediction falsified.** SPEC predicted 15–30 pp. See note below. |
| 8 | 2026-09-07 | D1 subject-ID, log-variance, no EA | 15 subj, leave-one-run-out | **0.975 ± 0.017** (chance 0.067) | Identity is trivially decodable, and across runs, so it is not a within-session artifact. |
| 9 | 2026-09-07 | D2 subject-ID, log-variance, EA | 15 subj, leave-one-run-out | **0.007 ± 0.008** (chance 0.067) | EA erases the covariance fingerprint completely — below chance, not merely at it. |
| 10 | 2026-09-07 | D3 subject-ID, relative log-PSD, no EA | 15 subj, leave-one-run-out | **0.948 ± 0.006** (chance 0.067) | Identity also lives in spectral shape. |
| 11 | 2026-09-07 | D4 subject-ID, relative log-PSD, EA | 15 subj, leave-one-run-out | **0.994 ± 0.006** (chance 0.067) | **The finding. Prediction confirmed.** EA leaves the spectral fingerprint fully intact — it is not even degraded. |

## Notes

### Things that surprised me

**B3 came back at zero, and the reason matters more than the number.** The spec
predicted random-trial-level k-fold would be inflated 15–30 pp over LOSO. It was
not: −1.2 pp and +3.4 pp. Subject leakage only inflates a score if the model has
the capacity to exploit it. Both pipelines here are linear and fit globally
across pooled subjects, and subject identity carries no information about the
left/right label — every subject contributes both classes, balanced. So there is
nothing for a linear model to gain from recognising the person. The 84–89 %
figures in the literature come from high-capacity models under trial-level
splits, which is a different claim than "trial-level splits inflate any model."
Follow-up: B4 tests this mechanism directly by adding capacity.

**D2 landed at 0.007, below the 0.067 chance level rather than at it.** EA does
not merely remove identity from covariance features, it makes subjects
systematically *mis*-identifiable across runs.

### Dead ends

- Absolute eigenvalue floor in the EA whitener (row 4). Produced a confident,
  wrong, publishable-looking result in the direction of the hypothesis, which is
  the worst kind of bug. Caught by asserting the aligned mean covariance is
  actually the identity — it was not.
- Parallel prefetch with `seq -w 1 32`, which pads to `S23` rather than `S023`.
  Every URL 404'd and curl without `-f` cheerfully saved the error pages as
  `.edf` files. 195 files "downloaded", all junk.

---

## Full grid, 30 subjects (2026-09-07)

| # | Experiment | Result | Conclusion |
|---|-----------|--------|------------|
| 12 | A2 ERD lateralisation, imagined | +0.203 ± 0.266, t=+4.12, p=0.0003, 24/30 subjects | Real at group level, and it confirms the T1/T2 mapping physiologically. 6/30 show the reverse. |
| 13 | A2 ERD lateralisation, executed | +0.149 ± 0.299, t=+2.68, p=0.012, 19/30 | Weaker than imagery, which is backwards from the usual claim. |
| 14 | A3 within-subject, csp_lda | 0.547 ± 0.183 [0.27–1.00] | The optimistic ceiling. |
| 15 | A3 within-subject, tangent_space | 0.567 ± 0.181 [0.27–0.98] | Barely above LOSO — per-subject calibration data buys almost nothing at 45 trials. |
| 16 | A4 mean per-subject 95% CI width | 27.9–28.0 pp | Any single-subject number here is uninformative. |
| 17 | B1/B2/B3 csp_lda | kfold 0.541, LOSO 0.534, gap **+0.7 pp** | Prediction (15–30 pp) falsified. |
| 18 | B1/B2/B3 tangent_space | kfold 0.619, LOSO **0.556 ± 0.084**, gap **+6.3 pp** | The honest headline number. |
| 19 | B4 capacity ladder | linear +6.3 pp, rbf_svm +1.4 pp, knn_1 +1.0 pp | **My follow-up hypothesis also falsified.** More capacity did not widen the gap. 1-NN in a 2080-dim tangent space is at chance (0.505) both ways — it cannot memorise usefully, so it cannot leak. |
| 20 | B5 window-overlap leakage, csp_lda | random-over-windows 0.537, by-trial 0.484, LOSO 0.545; window leak **+5.3 pp** | Real but small — 4 CSP components cannot memorise near-duplicates. |
| 20b | B5 window-overlap leakage, 1-NN on log-variance | random-over-windows **0.979**, by-trial 0.539, LOSO 0.517; window leak **+44.0 pp**, subject leak +2.2 pp | **The answer.** Overlapping windows split at random, plus a model that can look up near-duplicates, reproduces the published 84–89% range from a model that learned nothing. Subject pooling contributes ~2 pp; window overlap contributes ~44. |
| 21 | D1 subject-ID logvar, no EA | **0.960 ± 0.020** (chance 0.033) | |
| 22 | D2 subject-ID logvar, EA | **0.009 ± 0.010** | EA erases the covariance fingerprint entirely. |
| 23 | D3 subject-ID rel. log-PSD, no EA | **0.935 ± 0.017** | |
| 24 | D4 subject-ID rel. log-PSD, EA | **0.984 ± 0.012** | **The finding. Survives untouched — in fact slightly higher.** |
| 25 | D3b/D4b absolute log-PSD | 0.996 → 0.790 under EA | Mechanistic confirmation: EA normalises *power*, so absolute PSD drops. It does not touch spectral *shape*, so relative PSD does not. |
| 26 | D5 EA effect on task, csp_lda | **+10.7 pp ± 13.0**, helped 22/30, hurt 7/30 | Far above the 0–8 pp predicted. |
| 27 | D5 EA effect on task, tangent_space | +5.9 pp ± 9.7, helped 18/30, hurt 7/30 | |
| 28 | D6 per-subject EA delta | 7/30 hurt in both pipelines | Negative transfer is real, as predicted. |
| 29 | E1 executed → imagined | 0.577 ± 0.107 [0.38–0.84] | **Higher than imagery→imagery LOSO (0.556).** Training on the easier paradigm transfers better than training on the target one. |
| 30 | F1 fit gap, csp_lda | train 0.548 / test 0.534, **1.4 pp** | 4 CSP components cannot overfit. |
| 31 | F1 fit gap, tangent_space | train **0.998** / test 0.556, **44.2 pp** | Fits the training set perfectly and generalises at near chance. 2080 tangent-space features from ~1300 trials. This is the clearest overfitting evidence in the project. |
| 32 | F2 learning curve | 0.529 / 0.551 / 0.511 / 0.547 / 0.538 / 0.580 / **0.613** for n=3/6/10/15/20/25/29 | Noisy and flat through the middle, rising at the top. **Correction: I called this "flat" from a partial run that stopped at n=20; the full sweep rises.** Has not plateaued at 29, so more subjects would likely help. Individual points are resampling noise. |
| 32b | C1 permutation null | null mean 0.501, 95th pct 0.519, max 0.531; observed 0.556, **p=0.005** | Clears its own null, but by only ~4 pp over the 95th percentile. |
| 32c | C2 channel ablation | motor-only (9ch) **0.580**, all-64 0.556, non-motor (11ch) 0.521 | Signal is where physiology says. 9 motor channels beat all 64 — same over-parameterisation story as F1. |
| 32d | C3 EMG control | imagined 8–30 0.556 → 30–70 **0.526**; executed 8–30 0.625 → 30–70 **0.559** | Imagery collapses toward chance outside the ERD band; execution does not. Part of execution's advantage is muscle, not cortex. Prediction confirmed. |
| 32e | E2 imagined → executed | 0.590 ± 0.088 | Transfer works in both directions. |
| 32f | E3 pooled → imagined | 0.556 ± 0.093 | Pooling both paradigms buys nothing over imagery alone. |
| 33 | G1 shipped-model config | per-run EA 0.658 ± 0.165, per-subject EA 0.641, no EA 0.512 | The CLI's real number, and proof the capability is the alignment rather than the classifier. |

### Corrections made during the run

- **F2 "flat" was wrong.** I recorded the learning curve as flat from a partial
  run that had only reached n=20. The completed sweep rises to 0.613 at n=29.
  The conclusion changed from "more subjects do not help" to "the curve has not
  plateaued".
- **`write_rows` silently dropped columns.** It took its CSV fieldnames from the
  first row only, so in the heterogeneous F block every `n_train_subjects` and
  `n_components` value was written as an empty column. Caught when the figure
  script raised a KeyError. Fixed to take the union of all rows' keys, and block
  F was rerun. Any block writing mixed row types before this fix would have been
  affected; F is the only one that did.
- **Two hypotheses about leakage were falsified before the right one was found**
  (rows 17–19 and 20b). Recorded in order rather than rewritten to look like a
  straight line.
