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
