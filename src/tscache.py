"""Per-fold tangent-space features, computed once and reused.

A Riemannian mean of ~1300 64x64 covariances takes ~51 s. The LOSO folds are
identical across B2, B4, D5 and F1, and the covariance + tangent-space prefix is
unsupervised, so every one of those experiments was recomputing the same 30
means. This caches them.

Nothing about the evaluation changes: the prefix is still fit on each fold's
training trials only, and never sees the held-out subject.
"""

import hashlib
import os

import numpy as np
from joblib import Memory
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.model_selection import LeaveOneGroupOut

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "mne_data", "ts_cache")
_mem = Memory(CACHE, verbose=0)


def _key(X, groups):
    h = hashlib.md5()
    h.update(np.ascontiguousarray(X[::17, :, ::7]).tobytes())
    h.update(np.ascontiguousarray(groups).tobytes())
    h.update(str(X.shape).encode())
    return h.hexdigest()


@_mem.cache(ignore=["X", "groups"])
def _compute(key, X, groups, metric):
    folds = []
    for tr, te in LeaveOneGroupOut().split(X, np.zeros(len(X)), groups):
        cov = Covariances(estimator="oas")
        ts = TangentSpace(metric=metric)
        Ftr = ts.fit_transform(cov.fit_transform(X[tr]))
        Fte = ts.transform(cov.transform(X[te]))
        folds.append((tr, te, Ftr, Fte))
    return folds


def loso_ts_folds(X, groups, metric="riemann"):
    """[(train_idx, test_idx, F_train, F_test)] for each held-out subject."""
    return _compute(_key(X, groups), X, groups, metric)


def loso_with_features(folds, y, groups, make_clf):
    """Run a classifier over precomputed folds. Same output shape as evaluate.loso."""
    from src.evaluate import binomial_ci
    from sklearn.base import clone
    rows = []
    for tr, te, Ftr, Fte in folds:
        clf = clone(make_clf()).fit(Ftr, y[tr])
        pred = clf.predict(Fte)
        k, n = int((pred == y[te]).sum()), len(te)
        lo, hi = binomial_ci(k, n)
        rows.append({"subject": int(groups[te][0]), "acc": k / n, "n": n,
                     "ci_lo": lo, "ci_hi": hi,
                     "train_acc": float(clf.score(Ftr, y[tr]))})
    return rows
