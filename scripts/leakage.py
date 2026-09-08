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
from sklearn.model_selection import StratifiedKFold, GroupKFold

from src.data import load_dataset, good_subjects, IMAGINED_RUNS
from src.models import csp_lda
from src.probes import logvar_features
from src.report import write_rows
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def knn_logvar():
    """A model that can only answer by finding a near-identical training row.

    64 log-variance features, so unlike 1-NN in a 2080-dim tangent space this
    one is in a low enough dimension for nearest-neighbour to mean something.
    Paired with overlapping windows and a random split it is the exact recipe
    that inflates published numbers: the nearest neighbour of a test window is
    usually the window 0.25 s away from it, from the same trial, in training.
    """
    return Pipeline([
        ("logvar", FunctionTransformer(logvar_features)),
        ("sc", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=1)),
    ])


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

    rows = []
    for mname, pipe in (("csp_lda", csp_lda(4)), ("knn_1_logvar", knn_logvar())):
        print(f"\n-- {mname}")
        m, s = score(StratifiedKFold(5, shuffle=True, random_state=0).split(Xw, yw),
                     Xw, yw, pipe)
        print(f"B5 random 5-fold over WINDOWS      {m:.3f} +/- {s:.3f}   <- the leak")
        m2, s2 = score(GroupKFold(5).split(Xw, yw, tw), Xw, yw, pipe)
        print(f"B5 5-fold grouped by TRIAL         {m2:.3f} +/- {s2:.3f}")
        # 5-fold grouped by subject rather than leave-one-out: it answers the
        # same question (no person in both train and test) and costs 5 fits
        # instead of 106, which matters at 23k windows.
        m3, s3 = score(GroupKFold(5).split(Xw, yw, gw), Xw, yw, pipe)
        print(f"B5 5-fold grouped by SUBJECT       {m3:.3f} +/- {s3:.3f}")
        print(f"B5 window leakage  (random - by trial):    {(m-m2)*100:+.1f} pp")
        print(f"B5 subject leakage (by trial - by subject):{(m2-m3)*100:+.1f} pp")
        for k, (a, sd) in {"random_over_windows": (m, s), "grouped_by_trial": (m2, s2),
                           "grouped_by_subject": (m3, s3),
                           "gap_window_leak": (m - m2, float("nan")),
                           "gap_subject_leak": (m2 - m3, float("nan"))}.items():
            rows.append({"model": mname, "split": k, "acc": a, "sd": sd})
    write_rows("results/B5_window_leakage.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    main(ap.parse_args())
