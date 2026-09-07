"""Euclidean Alignment (He & Wu, IEEE TBME 67(2):399-410, 2020).

For each subject independently:
    R    = (1/N) sum_n Xn Xn^T      arithmetic mean of trial covariances
    X~n  = R^(-1/2) Xn

After this every subject's mean covariance is the identity, so the spatial
covariance structure that differs between people is whitened away.

Three properties make this usable here:
  - unsupervised: uses no labels from any domain, so applying it to a held-out
    subject inside LOSO is not leakage, and predict.py can legitimately apply it
    to an unseen EDF using only that file's own trials.
  - per-subject: computed from one person's data only, never pooled.
  - closed form: two operations, nothing fit.
"""

import numpy as np
from scipy.linalg import eigh


def ea_whitener(X, eps=1e-10):
    """R^(-1/2) for one subject's trials, X of shape (n, ch, t)."""
    covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    R = covs.mean(axis=0)
    # eigendecompose rather than fractional_matrix_power: CAR makes R rank
    # deficient by one, so an unfloored inverse square root blows up.
    w, V = eigh(R)
    w = np.maximum(w, eps)
    return (V * (w ** -0.5)) @ V.T


def align_subject(X, eps=1e-10):
    """Apply EA to one subject's trials."""
    return np.einsum("cd,ndt->nct", ea_whitener(X, eps), X)


def align_by_group(X, groups, eps=1e-10):
    """Apply EA independently within each subject.

    Done outside the CV loop on purpose and it is still not leakage: each
    subject's whitener sees only that subject's own unlabelled trials, which is
    exactly the information predict.py has at test time on a new EDF.
    """
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        out[m] = align_subject(X[m], eps)
    return out
