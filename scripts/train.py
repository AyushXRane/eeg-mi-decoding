"""Fit the shipped model on N subjects and save it for predict.py.

Trained on EA-aligned imagery from every training subject. EA is applied
per-subject and uses no labels, so the same transform can be applied to a new
EDF at test time without leaking anything.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np

from src.data import load_dataset, good_subjects, IMAGINED_RUNS, EXECUTED_RUNS
from src.align import align_by_group
from src.models import PIPELINES


def main(a):
    runs = {"imagined": IMAGINED_RUNS, "executed": EXECUTED_RUNS,
            "both": IMAGINED_RUNS + EXECUTED_RUNS}[a.runs]
    subs = good_subjects(a.subjects)
    ds = load_dataset(subs, runs)
    X = align_by_group(ds.X, ds.groups) if a.ea else ds.X

    pipe = PIPELINES[a.pipeline]().fit(X, ds.y)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    joblib.dump({"pipeline": pipe, "ea": a.ea, "l_freq": 8.0, "h_freq": 30.0,
                 "tmin": 0.5, "tmax": 3.5, "runs": a.runs,
                 "train_subjects": [int(s) for s in np.unique(ds.groups)],
                 "n_train_trials": int(len(ds.y)),
                 "classes": {0: "left_fist", 1: "right_fist"}}, a.out)
    print(f"saved {a.out}: {a.pipeline}, ea={a.ea}, "
          f"{len(np.unique(ds.groups))} subjects, {len(ds.y)} trials")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--runs", default="imagined", choices=["imagined", "executed", "both"])
    ap.add_argument("--pipeline", default="csp_lda", choices=list(PIPELINES))
    ap.add_argument("--ea", action="store_true", default=True)
    ap.add_argument("--no-ea", dest="ea", action="store_false")
    ap.add_argument("--out", default="models/csp_lda.joblib")
    main(ap.parse_args())
