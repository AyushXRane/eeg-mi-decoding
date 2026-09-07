"""The two pipelines. Both linear, both fit entirely inside the CV loop.

Rationale for staying linear: ~22 trials per class per subject and 64 channels.
A 216,714-pipeline MOABB benchmark found CSP and covariance tangent-space were
the only families competitive across datasets, scoring 0.604 +/- 0.178 and
0.605 +/- 0.174 on PhysionetMI within-session. Anything with more parameters
memorises at this sample size.
"""

from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def csp_lda(n_components=4):
    """Interpretable branch. CSP filters can be plotted and checked against
    sensorimotor cortex -- a spatial filter peaking at C3/C4 is evidence the
    model found the thing we claim it found.

    Shrinkage LDA specifically because covariance estimation is unstable with
    this few trials; ledoit_wolf regularisation inside CSP for the same reason.
    """
    return Pipeline([
        ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True,
                    norm_trace=False)),
        ("lda", LDA(solver="lsqr", shrinkage="auto")),
    ])


def tangent_space(C=1.0):
    """Performance branch. Covariances live on a curved manifold; the tangent
    space projection makes them a flat vector space a linear model can use.
    """
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("lr", LogisticRegression(C=C, max_iter=2000)),
    ])


PIPELINES = {"csp_lda": csp_lda, "tangent_space": tangent_space}


# --- Capacity ladder, for the leakage experiment (B4) only ---------------------
# B3 predicted a large gap between random trial-level k-fold and LOSO, and got
# roughly zero. The reason is that subject leakage can only inflate a score if
# the model can exploit it, and a global linear model cannot: subject identity
# carries no information about the left/right label, since every subject
# contributes both classes in balance.
#
# These three share one feature space (tangent-space projected covariances) and
# differ only in how much they can memorise. They exist to measure the
# leakage gap as a function of capacity. They are NOT performance models and
# none of them is the headline result.

def ts_knn(k=1):
    """Pure memorisation. 1-NN can only answer by finding a near-identical
    trial, which under a random split will often be another trial from the same
    person and the same run."""
    from sklearn.neighbors import KNeighborsClassifier
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("sc", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def ts_svm_rbf(C=10.0, gamma="scale"):
    """Intermediate capacity: non-linear but not a lookup table."""
    from sklearn.svm import SVC
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("sc", StandardScaler()),
        ("svm", SVC(C=C, gamma=gamma)),
    ])


CAPACITY_LADDER = {
    "linear_lr": tangent_space,     # cannot use subject identity
    "rbf_svm": ts_svm_rbf,          # can partially
    "knn_1": ts_knn,                # memorises outright
}
