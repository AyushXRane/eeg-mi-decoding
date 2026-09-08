"""G1: evaluate the shipped artifact in the configuration it actually ships in.

predict.py is handed one EDF, so it estimates the Euclidean Alignment whitener
from that run's ~15 trials. Training aligns per subject, pooling three runs.
That is a train/test mismatch, and reporting the grid's LOSO number as if it
described predict.py would be quietly wrong. This measures the real thing:
leave one subject out, then align the held-out subject the way predict.py has to.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.base import clone

from src.data import load_dataset, good_subjects, IMAGINED_RUNS
from src.align import align_by_group, align_subject
from src.models import PIPELINES
from src.evaluate import binomial_ci, summarise
from src.report import write_rows, line


def main(a):
    ds = load_dataset(good_subjects(a.subjects), IMAGINED_RUNS)
    subs = np.unique(ds.groups)
    rows = []

    for s in subs:
        tr = ds.groups != s
        te = ds.groups == s
        # Train exactly as train.py does: EA per subject over all their runs.
        Xtr = align_by_group(ds.X[tr], ds.groups[tr])
        mdl = clone(PIPELINES[a.pipeline]()).fit(Xtr, ds.y[tr])

        for mode in ("per_run", "per_subject", "none"):
            if mode == "per_run":
                Xte = np.concatenate([align_subject(ds.X[te & (ds.run == r)])
                                      for r in np.unique(ds.run[te])])
                yte = np.concatenate([ds.y[te & (ds.run == r)]
                                      for r in np.unique(ds.run[te])])
            elif mode == "per_subject":
                Xte, yte = align_subject(ds.X[te]), ds.y[te]
            else:
                Xte, yte = ds.X[te], ds.y[te]
            pred = mdl.predict(Xte)
            k, n = int((pred == yte).sum()), len(yte)
            lo, hi = binomial_ci(k, n)
            rows.append({"subject": int(s), "test_alignment": mode,
                         "acc": k / n, "n": n, "ci_lo": lo, "ci_hi": hi})

    write_rows(f"results/R1_model_capability.csv", rows)
    for mode in ("per_run", "per_subject", "none"):
        r = [x for x in rows if x["test_alignment"] == mode]
        print("G1 " + line(r, f"test-time EA = {mode}"))
        cross = sum(1 for x in r if x["ci_lo"] <= 0.5 <= x["ci_hi"])
        print(f"      per-subject CIs crossing chance: {cross}/{len(r)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=30)
    ap.add_argument("--pipeline", default="csp_lda", choices=list(PIPELINES))
    main(ap.parse_args())
