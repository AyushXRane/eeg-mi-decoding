"""CV schemes, confidence intervals, permutation nulls.

The distinction this module exists to enforce: which trials are allowed to share
a fold. Random k-fold lets trials from one person straddle train and test, which
is the wrong protocol and is included only so the gap against LOSO can be
measured.
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import beta
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut


def binomial_ci(k, n, alpha=0.05):
    """Clopper-Pearson interval. With ~15 test trials this is roughly +/-25pp,
    which is the point of reporting it."""
    if n == 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def loso(X, y, groups, pipe, return_train=False):
    """Leave-one-subject-out. The honest split: no trial from the test person is
    ever seen in training.

    Returns one row per held-out subject, each with its own binomial CI because
    a single subject's estimate rests on ~15 trials.
    """
    rows = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        p = clone(pipe).fit(X[tr], y[tr])
        pred = p.predict(X[te])
        k, n = int((pred == y[te]).sum()), len(te)
        lo, hi = binomial_ci(k, n)
        row = {"subject": int(groups[te][0]), "acc": k / n, "n": n,
               "ci_lo": lo, "ci_hi": hi}
        if return_train:
            row["train_acc"] = float(p.score(X[tr], y[tr]))
        rows.append(row)
    return rows


def loso_cross(X_tr, y_tr, g_tr, X_te, y_te, g_te, pipe):
    """Train on every other subject's data from one condition, test on the
    held-out subject's data from another. Used for execution -> imagery."""
    rows = []
    for s in np.unique(g_te):
        tr = g_tr != s
        te = g_te == s
        if te.sum() == 0 or tr.sum() == 0:
            continue
        p = clone(pipe).fit(X_tr[tr], y_tr[tr])
        pred = p.predict(X_te[te])
        k, n = int((pred == y_te[te]).sum()), int(te.sum())
        lo, hi = binomial_ci(k, n)
        rows.append({"subject": int(s), "acc": k / n, "n": n,
                     "ci_lo": lo, "ci_hi": hi})
    return rows


def random_kfold(X, y, pipe, n_splits=5, seed=0):
    """The deliberately wrong protocol: trials pooled across subjects and split
    at random, so the model sees the test subject during training. Reported only
    as the reference point that B3 measures the gap against.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = []
    for tr, te in cv.split(X, y):
        p = clone(pipe).fit(X[tr], y[tr])
        out.append(float(p.score(X[te], y[te])))
    return out


def within_subject(X, y, groups, pipe, n_splits=5, seed=0):
    """A3: stratified k-fold inside each subject separately.

    The optimistic ceiling -- the model gets labelled data from the very person
    it is tested on, which is the calibration burden a real BCI is trying to
    avoid. Every subject's estimate rests on ~9 test trials per fold, hence the
    CI on the pooled count.
    """
    rows = []
    for s in np.unique(groups):
        m = groups == s
        Xs, ys = X[m], y[m]
        if np.bincount(ys).min() < n_splits:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        k = n = 0
        for tr, te in cv.split(Xs, ys):
            pred = clone(pipe).fit(Xs[tr], ys[tr]).predict(Xs[te])
            k += int((pred == ys[te]).sum())
            n += len(te)
        lo, hi = binomial_ci(k, n)
        rows.append({"subject": int(s), "acc": k / n, "n": n, "ci_lo": lo, "ci_hi": hi})
    return rows


def _one_perm(X, y, groups, pipe, seed):
    rng = np.random.default_rng(seed)
    yp = y.copy()
    for g in np.unique(groups):
        m = groups == g
        yp[m] = rng.permutation(y[m])
    return float(np.mean([r["acc"] for r in loso(X, yp, groups, pipe)]))


def permutation_null(X, y, groups, pipe, n_perm=200, seed=0, n_jobs=-1):
    """Shuffle labels within each subject, then rerun the whole LOSO.

    Within-subject shuffling matters: a global shuffle would also destroy the
    subject-wise class balance, making the null easier than the real problem and
    the test too permissive. The null must also be built by rerunning the *same*
    CV -- a null from a different split structure tests a different question.
    """
    out = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_one_perm)(X, y, groups, pipe, seed + i) for i in range(n_perm))
    return np.array(out)


def summarise(rows, label=""):
    """Never a bare mean -- mean, sd and n_subjects, always together."""
    a = np.array([r["acc"] for r in rows])
    return {"label": label, "mean": float(a.mean()), "sd": float(a.std()),
            "min": float(a.min()), "max": float(a.max()), "n_subjects": len(a)}


def permutation_null_fast(X, y, groups, n_perm=200, seed=0, n_jobs=-1):
    """C1 for the tangent-space pipeline, without redoing unsupervised work.

    The pipeline splits cleanly into an unsupervised prefix (trial covariances,
    then the Riemannian mean and the tangent-space projection around it) and a
    supervised tail (logistic regression). Shuffling labels cannot change the
    prefix, so refitting it 200 times computes the same Riemannian mean 200
    times. Fitting it once per fold and permuting only the tail gives exactly
    the same null at ~1/200th the cost -- 30 Riemannian means instead of 6000.

    The prefix is still fit on training trials only, per fold, so nothing about
    the fold structure changes.
    """
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression

    folds = []
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        cov = Covariances(estimator="oas")
        ts = TangentSpace(metric="riemann")
        Ftr = ts.fit_transform(cov.fit_transform(X[tr]))
        Fte = ts.transform(cov.transform(X[te]))
        folds.append((Ftr, Fte, tr, te))

    def one(s):
        rng = np.random.default_rng(s)
        yp = y.copy()
        for g in np.unique(groups):
            m = groups == g
            yp[m] = rng.permutation(y[m])
        accs = []
        for Ftr, Fte, tr, te in folds:
            lr = LogisticRegression(C=1.0, max_iter=2000).fit(Ftr, yp[tr])
            accs.append(float((lr.predict(Fte) == yp[te]).mean()))
        return float(np.mean(accs))

    # The observed value must come from the same cached folds, or it is not
    # comparable to the null.
    observed = []
    for Ftr, Fte, tr, te in folds:
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Ftr, y[tr])
        observed.append(float((lr.predict(Fte) == y[te]).mean()))

    null = Parallel(n_jobs=n_jobs)(delayed(one)(seed + i) for i in range(n_perm))
    return np.array(null), float(np.mean(observed))
