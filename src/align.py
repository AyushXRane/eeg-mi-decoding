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


def ea_whitener(X, rtol=1e-6):
    """R^(-1/2) for one subject's trials, X of shape (n, ch, t).

    Common average referencing makes R singular -- the all-ones direction has
    zero variance by construction. A plain inverse square root multiplies that
    direction by 1/sqrt(eps), so numerical noise in an empty subspace comes back
    amplified ~1e5x and carries a subject signature of its own. Using the
    pseudo-inverse square root instead (drop eigenvalues below rtol * max)
    projects the null space out rather than exploding it.
    """
    covs = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    R = covs.mean(axis=0)
    w, V = eigh(R)
    inv = np.where(w > rtol * w.max(), w, np.inf) ** -0.5
    return (V * inv) @ V.T


def align_subject(X, rtol=1e-6):
    """Apply EA to one subject's trials."""
    return np.einsum("cd,ndt->nct", ea_whitener(X, rtol), X)


def align_by_group(X, groups, rtol=1e-6):
    """Apply EA independently within each subject.

    Done outside the CV loop on purpose and it is still not leakage: each
    subject's whitener sees only that subject's own unlabelled trials, which is
    exactly the information predict.py has at test time on a new EDF.
    """
    out = np.empty_like(X)
    for g in np.unique(groups):
        m = groups == g
        out[m] = align_subject(X[m], rtol)
    return out
