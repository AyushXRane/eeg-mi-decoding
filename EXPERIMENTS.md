# Experiment log

Five results, plus the things that went wrong on the way to them. Written when
each run finished, not reconstructed afterwards.

Data: PhysioNet EEGMMIDB. Identity probes use all **105** usable subjects
(109 minus S088/089/092/100, which have a nonstandard sampling rate). Accuracy
results use 30. Imagined runs 4/8/12 unless stated.

---

## The five results

| # | Question it answers | Experiment | Result |
|---|---|---|---|
| **R1** | What can the model do, and not do? | Leave-one-subject-out, three test-time alignment settings | **0.658** aligned · 0.641 per-subject · **0.512 unaligned (chance)**. Range 0.38–1.00; 16/30 individually above chance, 0 below, 14 inconclusive |
| **R2** | Is the number an artifact of how I split? | Trials cut into 87.5%-overlapping windows, then split three ways | **0.979** shuffling windows · 0.539 grouped by trial · 0.517 grouped by person. **+44 pp from window overlap, +2 pp from mixing people** |
| **R3** | What is the real baseline? | 200 within-subject label shuffles, whole LOSO rerun each time | null mean 0.501, **95th pct 0.519**, observed 0.556, **p = 0.005** |
| **R4** | Person or task? | Subject-ID probes, leave-one-run-out, 105 subjects, chance **0.0095** | covariance 0.946 → **0.026** under alignment; spectral shape 0.861 → **0.928**; absolute spectrum 0.972 → 0.714 |
| **R5** | Learned or memorised? | Train vs test accuracy on the same folds | CSP+LDA gap **1.4 pp** · tangent space **0.998 train / 0.556 test = 44 pp** |

**R4 replicates on the executed runs** (0.828 → 0.901 under alignment), so the
fingerprint is a property of the person and of the alignment method, not of
motor imagery.

**R4's third row is the mechanism.** Alignment is a *spatial* transform, so
anything in the *shape* of the spectrum passes through it untouched. Absolute
spectrum drops because it includes overall power, which alignment does normalise;
relative spectrum does not move.

---

## What went wrong

**A bug that faked my own hypothesis.** The first Euclidean Alignment
implementation floored the reference covariance's eigenvalues at an absolute
1e-10. Common average referencing makes that matrix singular by construction, so
the floor amplified an empty subspace ~10⁵× and the numerical noise living there
was itself subject-identifying. It produced 0.989 — a clean, confident result
pointing exactly where I wanted it to. Caught by asserting that the aligned mean
covariance actually equals the identity. It did not. After the fix: 0.009.

**Two predictions falsified.** I expected mixing subjects across train and test
to inflate accuracy by 15–30 points. It was ~1 point. I then guessed that model
capacity explained the null result — that a model powerful enough to memorise
would show the gap. Also wrong: 1-nearest-neighbour had the *smallest* gap,
because in 2080 dimensions it is at chance either way. R2 is where that hunt
finally landed, and it is a better answer than the prediction was.

**A claim I corrected.** I described the learning curve as flat from a partial
run that had only reached 20 training subjects. The completed sweep rises to
0.613 at 29, so the conclusion changed from "more subjects do not help" to "the
curve has not plateaued."

**A silent data bug.** The CSV writer took its column names from the first row
only, so in a block with heterogeneous rows some columns were written empty.
Caught when a plotting script raised a KeyError.

**A download bug.** The parallel prefetch padded subject numbers to two digits
(`S23` instead of `S023`). Every URL 404'd, and curl without `-f` saved the error
pages as `.edf` files. 195 files "downloaded", all junk.

---

## Cut

A further ~20 experiments were run and then removed to keep this defensible:
ERD lateralisation, within-subject ceiling, channel ablation, a 30–70 Hz EMG
control, execution↔imagery transfer in both directions, per-subject alignment
deltas, a learning curve, and CSP component sweeps. They are recoverable from git
history if needed. The five above are the ones that carry the argument.
