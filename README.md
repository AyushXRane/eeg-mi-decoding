# Left vs right fist decoding on EEGMMIDB — and what the number actually means

NT@B FA26 Software Division recruitment project.

**Headline: cross-subject imagery decoding sits at ~0.61 and every individual
subject's confidence interval crosses chance. Meanwhile subject identity is
decodable from the same preprocessed data at 0.99 against a chance level of
0.033. Euclidean Alignment — recommended in the literature as standard
preprocessing for cross-subject models — destroys the covariance half of that
identity signal and leaves the spectral half completely untouched.**

That last sentence is the finding. Everything else is the evidence for it.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./scripts/fetch.sh 30 "3 4 7 8 11 12" 8   # ~6 min, parallel; optional but much faster
python scripts/verify_labels.py           # A1: check T1/T2 before trusting anything
python scripts/gonogo.py --subjects 15    # the four numbers that set the framing
python scripts/grid.py all --subjects 30  # the full experiment grid -> results/*.csv
python scripts/figures.py                 # -> results/*.png
```

### Prediction CLI

```bash
python scripts/train.py --subjects 30 --out models/csp_lda.joblib
python scripts/predict.py --edf ~/mne_data/MNE-eegbci-data/files/eegmmidb/1.0.0/S042/S042R04.edf
```

Prints one predicted label per trial. It applies the identical preprocessing
chain, then Euclidean Alignment computed from **only that file's own trials** —
legitimate because EA is unsupervised, and it is the only adaptation a real BCI
could perform before a new user has produced a single label. It warns if handed
a run outside {3,4,7,8,11,12}, refuses recordings at the wrong sampling rate, and
skips EA rather than estimating a whitener from fewer than four trials.

---

## The data, and what I checked before trusting it

PhysioNet EEGMMIDB: 109 subjects, 64 channels, 160 Hz, BCI2000. I used the first
30 subjects that pass QC. Thirty is enough for a LOSO estimate whose standard
error is small relative to the between-subject spread — that spread is ~0.09, so
30 subjects put the standard error near 0.016, and the conclusions here turn on
gaps of 5–40 points, not 2. More subjects buy download time, not resolution.

**Excluded by construction:** S088, S089, S092, S100 — nonstandard sampling rate
or trial duration, so their epochs come out the wrong shape.

Some published curations additionally drop S038, S104 and S106 for annotation
problems. These load cleanly here and stay in. **That different papers exclude
different subjects from the same public dataset, with no shared standard, is
itself worth noticing** — it is a small free parameter that moves reported
accuracies and is almost never stated in an abstract.

### The T1/T2 trap

The task description says the meaning of T1 and T2 depends on the run and tells
you to check. I checked, and the check has a sharper edge than expected:

| | T0 | T1 | T2 |
|---|---|---|---|
| runs 3/7/11 (L/R fist, executed) | 45 @ 4.1–4.2 s | 22–23 @ 4.10 s | 22–23 @ 4.10 s |
| runs 5/9/13 (fists/feet, executed) | 45 @ 4.1–4.2 s | 22–23 @ 4.10 s | 22–23 @ 4.10 s |

**The annotation structure is identical.** Counts, durations, cadence — there is
nothing inside the file that distinguishes "left fist vs right fist" from "both
fists vs both feet". Only the run number in the filename does. Mislabel this and
you get a trained model, a plausible accuracy, and no error message anywhere.

Because the filename is a weak thing to stake a project on, `verify_labels.py`
also checks the mapping physiologically: left- vs right-hand movement should
desynchronise mu over the *contralateral* hemisphere, so C3 and C4 should move in
opposite directions in the L/R fist runs. See A2 below — they do.

### Preprocessing

```
raw EDF (64ch, 160 Hz)
  -> eegbci.standardize()            strip trailing dots from channel names
  -> set_montage(standard_1005)
  -> bandpass 8-30 Hz, zero-phase FIR
  -> common average reference
  -> epoch 0.5-3.5 s post-cue
  -> [Euclidean Alignment]           per-subject, unsupervised, ablation switch
  -> CSP(4) -> shrinkage LDA   |  Covariances(oas) -> TangentSpace -> LogReg
  -> LeaveOneGroupOut(groups=subject)
```

- **8–30 Hz** is mu plus beta, where event-related desynchronisation lives.
  Outside that band there is no ERD to find. It also removes 60 Hz line noise and
  sub-4 Hz drift for free, which is most of the eye-blink energy — that is why
  there is no ICA step, and C2 below is the check on whether that was enough.
- **CAR** shrinks the between-subject magnitude differences a shared reference
  introduces, which matters for anything cross-subject.
- **0.5–3.5 s** skips the onset evoked transient and ends before the next trial.
- Filtering and CAR happen on **continuous** data, before epoching, so filter
  edge effects do not eat into the analysis window.
- Every model step — CSP, scalers, LDA, the tangent-space projection — is inside
  an `sklearn.Pipeline` and refit on each fold's training set only.

---

## How I decided what train and test share

This is the question the whole evaluation turns on, so it gets stated plainly.

**The split is leave-one-subject-out.** A model is trained on 29 people and
tested on the 30th, whose data it has never seen in any form. This matches the
deployment situation the brief describes: a new person sits down and the model
must work with zero labels from them.

Two other splits appear, both as reference points rather than results:

- **Random trial-level 5-fold** pools every trial from every subject and splits
  at random, so ~80% of the test subject's own trials are in training. This is
  the protocol much of the published literature uses and it is the wrong one.
- **Within-subject k-fold** trains and tests inside one person. That is the
  optimistic ceiling and it corresponds to a BCI that makes the user sit through
  a labelled calibration session first.

The subject-ID probes use a third split — **leave-one-run-out**. Every subject
necessarily appears in both train and test, because subject identity *is* the
label, but train and test trials come from different recording runs. Without
that, a high identity score would just mean two halves of one continuous
recording look alike.
