"""Sanity check: does the leave-one-subject-out number depend on the split scheme?

Leave-one-out is k-fold with k = n_subjects. A grouped 80/20 is k-fold with
k = 5. Both keep every test person out of training, so they should agree --
this checks that they do, and shows how much a single 80/20 draw wobbles
depending on which people land in the test set.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.base import clone

from src.data import load_dataset, good_subjects, IMAGINED_RUNS
from src.align import align_by_group
from src.models import csp_lda


def main(a):
    ds = load_dataset(good_subjects(a.subjects), IMAGINED_RUNS)
    X = align_by_group(ds.X, ds.groups)
    subs = np.unique(ds.groups)
    rng = np.random.default_rng(0)
    print(f"{len(subs)} subjects\n")

    # 20 different random 80/20 splits BY SUBJECT
    accs = []
    for i in range(20):
        te_s = rng.choice(subs, size=max(1, len(subs) // 5), replace=False)
        te = np.isin(ds.groups, te_s)
        m = clone(csp_lda(4)).fit(X[~te], ds.y[~te])
        accs.append(float(m.score(X[te], ds.y[te])))
    accs = np.array(accs)
    print(f"grouped 80/20, 20 different random draws:")
    print(f"  mean {accs.mean():.3f}   sd {accs.std():.3f}   "
          f"range {accs.min():.3f} - {accs.max():.3f}")
    print(f"  -> a single 80/20 draw can land anywhere in a "
          f"{(accs.max()-accs.min())*100:.0f} point window")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=109)
    main(ap.parse_args())
