"""B5: where the leakage in the EEG literature actually comes from.

B3 and B4 both came back near zero. Pooling subjects and splitting trials at
random barely inflates anything here, and adding model capacity does not change
that. So the large gaps reported elsewhere must come from somewhere else.

The usual suspect is windowing. Many EEG pipelines cut each trial into several
overlapping windows to multiply their sample count, then split those windows at
random. Two windows from the same trial overlapping by 75% are very nearly the
same data, so the test set contains near-duplicates of training rows. That is a
different and much more powerful leak than merely sharing a subject.

This builds exactly that setup and evaluates it three ways:
  random over windows  -- windows from one trial straddle train and test
  grouped by trial     -- windows stay with their trial, subjects still mixed
  grouped by subject   -- the honest split
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GroupKFold, LeaveOneGroupOut

from src.data import load_dataset, good_subjects, IMAGINED_RUNS
from src.models import csp_lda
from src.report import write_rows


def windowise(X, y, groups, win=320, step=40):
    """Cut each 3 s epoch into overlapping windows, as a data-augmentation step
    would. win=320 samples = 2.0 s, step=40 = 0.25 s, so adjacent windows share
    87.5% of their samples."""
    Xs, ys, gs, ts = [], [], [], []
    starts = range(0, X.shape[-1] - win + 1, step)
    for i in range(len(X)):
        for s0 in starts:
            Xs.append(X[i, :, s0:s0 + win])
            ys.append(y[i])
            gs.append(groups[i])
            ts.append(i)                      # trial id -- the thing that leaks
    return (np.asarray(Xs), np.asarray(ys), np.asarray(gs), np.asarray(ts))


def score(cv_iter, X, y, pipe):
    accs = []
    for tr, te in cv_iter:
        p = clone(pipe).fit(X[tr], y[tr])
        accs.append(float(p.score(X[te], y[te])))
    return float(np.mean(accs)), float(np.std(accs))


def main(a):
    ds = load_dataset(good_subjects(a.subjects), IMAGINED_RUNS)
    Xw, yw, gw, tw = windowise(ds.X, ds.y, ds.groups)
    print(f"{len(ds.X)} trials -> {len(Xw)} windows "
          f"({len(Xw)//len(ds.X)} per trial, 87.5% overlap)")

    pipe = csp_lda(4)
    rows = []

    m, s = score(StratifiedKFold(5, shuffle=True, random_state=0).split(Xw, yw),
                 Xw, yw, pipe)
    print(f"B5 random 5-fold over WINDOWS      {m:.3f} +/- {s:.3f}   <- the leak")
    rows.append({"split": "random_over_windows", "acc": m, "sd": s})

    m2, s2 = score(GroupKFold(5).split(Xw, yw, tw), Xw, yw, pipe)
    print(f"B5 5-fold grouped by TRIAL         {m2:.3f} +/- {s2:.3f}")
    rows.append({"split": "grouped_by_trial", "acc": m2, "sd": s2})

    m3, s3 = score(LeaveOneGroupOut().split(Xw, yw, gw), Xw, yw, pipe)
    print(f"B5 leave-one-SUBJECT-out           {m3:.3f} +/- {s3:.3f}")
    rows.append({"split": "grouped_by_subject", "acc": m3, "sd": s3})

    print(f"\nB5 window leakage  (random - by trial):   {(m-m2)*100:+.1f} pp")
    print(f"B5 subject leakage (by trial - by subject): {(m2-m3)*100:+.1f} pp")
    rows.append({"split": "gap_window_leak", "acc": m - m2, "sd": float("nan")})
    rows.append({"split": "gap_subject_leak", "acc": m2 - m3, "sd": float("nan")})
    write_rows("results/B5_window_leakage.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    main(ap.parse_args())
