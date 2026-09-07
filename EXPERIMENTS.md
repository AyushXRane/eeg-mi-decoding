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
