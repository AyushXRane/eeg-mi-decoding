# The five results — everything you need to defend

The full attempt log is in `EXPERIMENTS.md`. This file is the story: one result
per question the brief actually asks. If you can explain these five, you can
explain the project.

---

## 0. The pipeline (30 seconds, not a result)

Raw EDF → bandpass **8–30 Hz** → common average reference → cut into **3-second
trials** → **Euclidean Alignment** → **CSP (4 filters) → shrinkage LDA**.

Why 8–30 Hz: that is the mu and beta band, where the effect lives — imagining a
movement desynchronises motor cortex and band power drops on the opposite side.
Outside that band there is nothing to find. It also removes 60 Hz line noise and
slow drift for free, which is most of the eye-blink energy — that is why there is
no ICA step.

Tested on **all 106 usable subjects**, evaluated **leave-one-subject-out**:
train on 105 people, test on the 106th, repeat 106 times so every person is the
test set exactly once. Every number below is that.

*(Sanity check: 20 random 80/20 grouped splits average 0.686 — the same answer —
but individual draws range 0.633 to 0.739. Leave-one-out gives that number
without depending on which people you happened to draw.)*

---

## 1. What the model can and cannot do  →  **0.695**

All 106 subjects, leave-one-subject-out.

| test-time alignment | accuracy | individuals beating chance |
|---|---|---|
| aligned (what the CLI does) | **0.695 ± 0.157** | **60 / 106** |
| aligned per subject | 0.688 ± 0.148 | 53 / 106 |
| not aligned | **0.598 ± 0.107** | 29 / 106 |

Range across people: 0.36 to 1.00.

**The line to say:** alignment is worth about 10 points — 0.695 with it, 0.598
without. It is the single largest component in the pipeline, larger than the
choice of classifier, which is worth about 2.

**A correction worth telling them about.** At 30 subjects the unaligned model
scored 0.512 — dead chance — and I was ready to claim the classifier does nothing
without alignment. At 106 subjects it scores 0.598. With 105 training subjects
instead of 29 the model is robust enough to survive the mismatch, so the dramatic
collapse was partly a small-sample effect. This is the project's own lesson
applied to itself: a striking number at n=30 was partly noise.

---

## 2. Is the number an artifact of how I measured it?  →  **0.979 from nothing**

Most EEG papers chop each trial into overlapping windows to multiply their sample
count, then shuffle those windows into train and test. Two windows from the same
trial overlapping 87.5% are nearly the same data.

I built exactly that setup and ran a 1-nearest-neighbour classifier:

| split | accuracy |
|---|---|
| shuffled **windows** | **0.978** |
| grouped by **trial** | 0.529 |
| grouped by **person** | 0.507 |

**0.978 from a model that learned nothing about motor imagery** — its nearest
neighbour is just the window 0.25 s away from the same trial. That reproduces the
84–89% range published on this dataset.

**The line to say:** **44.8 points** of that inflation is window overlap, only
**2.3 points** is mixing people. So the standard advice — "use subject-wise
splits" — is necessary and nowhere near sufficient. You have to group by trial.

---

## 3. What is the real baseline?  →  **0.519, not 0.50**

I shuffled the labels within each person and reran the entire evaluation 200
times.

| | |
|---|---|
| null mean | 0.500 |
| **null 95th percentile** | **0.511** |
| null maximum over 200 shuffles | 0.519 |
| my result | **0.593** |
| p | **0.005** |

**The line to say:** chance is 0.50, but the number you actually have to beat is
0.511 — and no shuffle in 200 ever exceeded 0.519. My result clears that by
eight points.

---

## 4. Person or task?  →  **0.93 vs 0.66**

Same preprocessed data, two questions asked of it. Chance for "who is this?" is
1/105 = **0.0095**.

| | no alignment | **after alignment** |
|---|---|---|
| identity, from covariance | 0.947 | **0.027** |
| **identity, from spectrum shape** | 0.865 | **0.926** |
| identity, from absolute spectrum | 0.973 | 0.710 |

Replicated on the executed runs: 0.827 → **0.905**. Not an imagery artifact.

Alignment completely erases one fingerprint and leaves the other **untouched**.

**Why**, and this is the part to know cold: alignment is a *spatial* transform —
it mixes channels. Anything encoded in the *shape* of the frequency spectrum
passes straight through it. The third row proves it: absolute spectrum drops
(0.972 → 0.714) because it includes overall power, which alignment does
normalise. Pure shape does not move.

**The line to say:** on the same data, the machine identifies *who you are* at
0.93 against a 1-in-106 chance level, and *what you imagined* at 0.70. The
field's standard correction for this removes half the problem.

---

## 5. Learned or memorised?  →  **0.998 vs 0.556**

| model | train | test | gap |
|---|---|---|---|
| CSP + LDA (4 filters) | 0.614 | 0.609 | **0.5 pp** |
| tangent space (2080 features) | **0.882** | 0.593 | **28.9 pp** |

**The line to say:** this is why there is no neural network here. My *linear*
model already memorises the training set almost perfectly with 2080 parameters
and 1300 trials. A ConvNet would make that worse and harder to see. The simple
model — four spatial filters — is both more accurate after alignment and honest
about what it knows.

---

# The three required extras

**One design decision.** Linear over deep learning. The real alternative was
EEGNet. Rejected on sample size, and result #5 is the proof it was right rather
than merely cautious.

**Weakest point.** Result #4 shows identity *survives* alignment. It does **not**
show that surviving identity is what caps accuracy. Different claims; I only
measured the first. *What would make me wrong:* if the spectral fingerprint comes
from electrode impedance rather than neurophysiology, it is a finding about
hardware. My recordings are same-session, same cap, so I cannot rule that out.

**Something you did not ask about.** Result #2 — the window leak — or the fact
that the fists/feet runs have **identical** annotation structure to the
left/right runs, so nothing inside the file tells you which task it is. Only the
filename does.

---

# If they push further

Everything below was run and is in `EXPERIMENTS.md`. Mention only if asked.

- Two predictions of mine were **falsified** and I reported them: I expected
  subject-mixing to inflate results 15–30 points (it was ~1), then guessed model
  capacity explained it (it did not). Result #2 is where that hunt ended up.
- I shipped a **bug that faked my own hypothesis** — a numerical error made
  identity look like it survived when it did not. Caught by asserting the
  alignment actually did what it claims. Fixed; 0.989 → 0.009.
- Execution-trained models decode imagery **better** than imagery-trained ones
  (0.577 vs 0.556) — but that advantage vanishes once you align, so it was
  covariance structure all along.
- Execution's "stronger signal" is partly **muscle**: at 30–70 Hz, where there is
  no brain effect to find, imagery collapses to chance and execution does not.
- Signal is where physiology says: 9 motor channels (0.580) beat all 64 (0.556)
  beat frontal/occipital (0.521).

---

## R6 — Imagery or execution? (the brief suggests asking this)

The brief says you may use executed runs, imagined runs, or both, and that
asking whether a model trained on one transfers to the other is "fair game and
possibly interesting." So I asked.

**Imagery is the headline** for two reasons. It is the BCI-relevant problem — a
paralysed user cannot execute a movement — and execution's famous "stronger
signal" is partly not brain at all. I tested that: refit the whole pipeline at
30–70 Hz, a band where the real neural effect does not exist.

| | 8–30 Hz (real effect lives here) | 30–70 Hz (nothing should survive) |
|---|---|---|
| imagery | 0.556 | **0.526 — collapses to chance** |
| execution | 0.625 | **0.559 — still decodable** |

Execution stays decodable in a band where there is no brain signal to decode.
That is muscle activity, which is broadband and sits closer to the electrodes
than cortex does. Part of execution's advantage is not neural.

**But I still trained across them both ways:**

All 106 subjects, with alignment:

| | accuracy |
|---|---|
| train imagery → test imagery | **0.688** |
| train execution → test imagery | **0.674** |
| train imagery → test execution | 0.715 |
| train on both → test imagery | 0.687 |

**Training on execution decodes imagery almost as well as training on imagery
itself** — 0.674 vs 0.688 — despite having to cross two gaps at once, a new
person *and* a different task.

At 30 subjects and *without* alignment, execution actually won (0.577 vs 0.556).
Once you align, that advantage disappears. So execution was never teaching the
model something extra about motor imagery — it was giving better estimates of
the between-person differences that alignment already removes. Two routes to the
same correction.

Practically: to calibrate a new user you could have them **actually move**,
which is easier and less error-prone than coaching imagery, and lose almost
nothing.

**The line to say:** execution transfers to imagery surprisingly well, but only
because it is a backdoor to the same fix alignment does directly. And its extra
signal is partly muscle, which is why imagery is the number I defend.
