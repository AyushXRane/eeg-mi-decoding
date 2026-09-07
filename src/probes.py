"""Probes for what the pipeline encodes besides the task.

Everything here runs on the *same* preprocessed epochs the task classifier sees,
so the claims are about the actual pipeline rather than about the raw data.
"""

import numpy as np
from scipy.signal import welch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SFREQ = 160.0

# C3/C4 are directly over left/right hand motor cortex; Cz sits between them.
MOTOR = ["C3", "Cz", "C4", "C1", "C2", "Cp3", "Cp4", "Fc3", "Fc4"]
NON_MOTOR = ["Fp1", "Fp2", "Af7", "Af8", "F7", "F8", "O1", "O2", "Oz", "Po7", "Po8"]


def logvar_features(X):
    """Log variance per channel -- the diagonal of the trial covariance.

    This is the feature family EA is designed to normalise, so it is the right
    thing to probe with for D1/D2.
    """
    return np.log(X.var(axis=-1) + 1e-12)


def psd_features(X, fmin=8.0, fmax=30.0, relative=True, sfreq=SFREQ):
    """Log band power per channel per 1 Hz bin, flattened.

    `relative` divides each channel's spectrum by its own total power before the
    log, which strips the broadband scale and leaves the *shape* of the
    spectrum. That distinction is the whole of D4: EA renormalises power, it
    does not change spectral shape.
    """
    f, P = welch(X, fs=sfreq, nperseg=int(sfreq), axis=-1)
    m = (f >= fmin) & (f <= fmax)
    P = P[..., m]
    if relative:
        P = P / (P.sum(axis=-1, keepdims=True) + 1e-30)
    return np.log(P + 1e-30).reshape(len(X), -1)


def _probe_clf():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=3000, C=1.0))])


def subject_id_probe(X, groups, runs, feature_fn, clf=None):
    """How well can subject identity be read out, generalising across runs?

    Leave-one-run-out rather than a random trial split. Every subject appears in
    train and test by construction -- that is what identity decoding means -- but
    train and test trials come from different recording runs, so a score here
    cannot be explained by two halves of one continuous recording looking alike.

    Chance is 1/n_subjects, not 0.5.
    """
    F = feature_fn(X)
    clf = clf or _probe_clf()
    accs, ns = [], []
    for r in np.unique(runs):
        te = runs == r
        tr = ~te
        if len(np.unique(groups[tr])) < 2:
            continue
        p = clone(clf).fit(F[tr], groups[tr])
        accs.append(float((p.predict(F[te]) == groups[te]).mean()))
        ns.append(int(te.sum()))
    return {"acc": float(np.mean(accs)), "sd": float(np.std(accs)),
            "per_fold": accs, "n_test": ns,
            "chance": 1.0 / len(np.unique(groups)),
            "n_subjects": int(len(np.unique(groups)))}


def pick_channels(ch_names, wanted):
    """Indices of the requested channels, case-insensitively."""
    low = [c.lower() for c in ch_names]
    return [low.index(w.lower()) for w in wanted if w.lower() in low]


def erd_check(X, y, ch_names, sfreq=SFREQ):
    """A2: does the physiological effect exist at all?

    Mu-band power at C3 and C4 for left vs right imagery. Contralateral
    desynchronisation means right-hand imagery should drop power at C3 more than
    at C4, and vice versa. Reported per subject by the caller because it will
    not hold for everyone.
    """
    idx = pick_channels(ch_names, ["C3", "C4"])
    if len(idx) < 2:
        return None
    f, P = welch(X[:, idx, :], fs=sfreq, nperseg=int(sfreq), axis=-1)
    mu = (f >= 8) & (f <= 13)
    power = np.log(P[..., mu].mean(axis=-1) + 1e-30)   # (n_trials, 2)
    left, right = power[y == 0], power[y == 1]
    # Positive lateralisation index = the expected contralateral pattern.
    return {"C3_left": float(left[:, 0].mean()), "C3_right": float(right[:, 0].mean()),
            "C4_left": float(left[:, 1].mean()), "C4_right": float(right[:, 1].mean()),
            "lat_index": float((right[:, 0].mean() - left[:, 0].mean())
                               - (right[:, 1].mean() - left[:, 1].mean()))}
