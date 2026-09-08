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

Tested on **105 subjects**, evaluated **leave-one-subject-out**: train on 104
people, test on the 105th, repeat 105 times so every person is the test set once.

---

## 1. What the model can and cannot do  →  **0.658**

| test-time alignment | accuracy |
|---|---|
| aligned (what the CLI does) | **0.658** |
| not aligned | **0.512 — chance** |

Range across people: 0.38 to 1.00. Sixteen of thirty individually beat chance,
none is below it, fourteen are inconclusive.

**The line to say:** almost none of this is the classifier. Alignment is worth
~11 points; the choice of model is worth ~2. Take the alignment away and the same
trained model is at chance.

---

## 2. Is the number an artifact of how I measured it?  →  **0.979 from nothing**

Most EEG papers chop each trial into overlapping windows to multiply their sample
count, then shuffle those windows into train and test. Two windows from the same
trial overlapping 87.5% are nearly the same data.

I built exactly that setup and ran a 1-nearest-neighbour classifier:

| split | accuracy |
|---|---|
| shuffled **windows** | **0.979** |
| grouped by **trial** | 0.539 |
| grouped by **person** | 0.517 |

**0.979 from a model that learned nothing about motor imagery** — its nearest
neighbour is just the window 0.25 s away from the same trial. That reproduces the
84–89% range published on this dataset.

**The line to say:** 44 points of that inflation is window overlap, 2 points is
mixing people. So the standard advice — "use subject-wise splits" — is necessary
and nowhere near sufficient.

---

## 3. What is the real baseline?  →  **0.519, not 0.50**

I shuffled the labels within each person and reran the entire evaluation 200
times.

| | |
|---|---|
| null mean | 0.501 |
| **null 95th percentile** | **0.519** |
| my result | 0.556 |
| p | **0.005** |

**The line to say:** chance is 0.50, but the number you have to beat is 0.519.
My result clears it, by about four points. Not a landslide.

---

## 4. Person or task?  →  **0.93 vs 0.66**

Same preprocessed data, two questions asked of it. Chance for "who is this?" is
1/105 = **0.0095**.

| | no alignment | **after alignment** |
|---|---|---|
| identity, from covariance | 0.946 | **0.026** |
| **identity, from spectrum shape** | 0.861 | **0.928** |
| identity, from absolute spectrum | 0.972 | 0.714 |

Alignment completely erases one fingerprint and leaves the other **untouched**.

**Why**, and this is the part to know cold: alignment is a *spatial* transform —
it mixes channels. Anything encoded in the *shape* of the frequency spectrum
passes straight through it. The third row proves it: absolute spectrum drops
(0.972 → 0.714) because it includes overall power, which alignment does
normalise. Pure shape does not move.

**The line to say:** on the same data, the machine identifies *who you are* at
0.93 against a 1-in-105 chance level, and *what you imagined* at 0.66. The
field's standard correction for this removes half the problem.

---

## 5. Learned or memorised?  →  **0.998 vs 0.556**

| model | train | test | gap |
|---|---|---|---|
| CSP + LDA (4 filters) | 0.548 | 0.534 | **1.4 pp** |
| tangent space (2080 features) | **0.998** | 0.556 | **44 pp** |

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

| | without alignment | with alignment |
|---|---|---|
| train imagery → test imagery | 0.556 | 0.641 |
| **train execution → test imagery** | **0.577** | 0.641 |
| train imagery → test execution | 0.590 | 0.684 |
| train on both → test imagery | 0.556 | 0.656 |

**Without alignment, training on execution beats training on imagery itself**
(0.577 vs 0.556) — even though it has to cross two gaps at once, a new person
*and* a different task. Executed trials give a cleaner signal, so the spatial
filters are better estimated and they transfer.

**With alignment that advantage disappears completely** — 0.641 either way. So
execution was never teaching the model something extra about motor imagery. It
was giving better estimates of the between-person differences that alignment
already removes. Two routes to the same correction.

**The line to say:** execution transfers to imagery surprisingly well, but only
because it is a backdoor to the same fix alignment does directly. And its extra
signal is partly muscle, which is why imagery is the number I defend.
