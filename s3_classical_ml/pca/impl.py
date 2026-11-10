"""PCA as a scikit-learn-style model (fit / transform / inverse_transform).

The maths lives in s2_math/pca; this wraps it with the conveniences used in practice:
choosing k by explained variance, whitening, and a pipeline-friendly API.
"""

from __future__ import annotations

import numpy as np

from s2_math.pca.impl import pca_svd


class PCA:
    """n_components may be an int (keep that many) or a float in (0, 1)
    (keep the fewest components explaining at least that fraction of variance)."""

    def __init__(self, n_components: int | float | None = None, whiten: bool = False) -> None:
        self.n_components, self.whiten = n_components, whiten

    def fit(self, X: np.ndarray) -> PCA:
        X = np.asarray(X, dtype=float)
        full = min(X.shape)
        components, variance, mean = pca_svd(X, full)
        ratio = variance / X.var(axis=0, ddof=1).sum()
        if self.n_components is None:
            k = full
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            # smallest k whose cumulative ratio reaches the target
            k = int(np.searchsorted(np.cumsum(ratio), self.n_components - 1e-12) + 1)
        else:
            k = int(self.n_components)
        self.n_components_ = k
        self.components_ = components[:k]
        self.explained_variance_ = variance[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.mean_ = mean
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T
        if self.whiten:  # rescale every component to unit variance
            Z = Z / np.sqrt(self.explained_variance_)
        return Z

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_
