"""PCA two ways: eigendecomposition of the covariance matrix, and SVD of the data.

Both return the same thing; seeing why is the point of this module.
"""

from __future__ import annotations

import numpy as np


def _fix_signs(components: np.ndarray) -> np.ndarray:
    """An eigenvector times -1 is still an eigenvector. Pick the sign that makes
    each component's largest-magnitude entry positive so results are reproducible."""
    idx = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(len(components)), idx])
    signs[signs == 0] = 1
    return components * signs[:, None]


def pca_eig(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via the covariance matrix.

    1. Centre the data (subtract each feature's mean).
    2. Covariance C = X_c^T X_c / (n - 1)   (d x d, symmetric).
    3. Eigenvectors of C are the principal directions; eigenvalues are the
       variance of the data along each direction.
    4. Keep the k directions with the largest eigenvalues.

    Returns (components [k, d], explained_variance [k], mean [d]).
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = Xc.T @ Xc / (len(X) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # eigh: for symmetric matrices, ascending order
    order = np.argsort(eigvals)[::-1][:k]
    components = _fix_signs(eigvecs[:, order].T)
    return components, eigvals[order], mean


def pca_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via SVD of the centred data: X_c = U S V^T.

    Then C = X_c^T X_c / (n-1) = V (S^2 / (n-1)) V^T, which IS the
    eigendecomposition of C. So the rows of V^T are the principal directions
    and S^2 / (n - 1) are the variances. No covariance matrix is ever formed,
    which is more accurate (no squaring) and cheaper when d is large.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mean, full_matrices=False)  # S is already descending
    return _fix_signs(Vt[:k]), S[:k] ** 2 / (len(X) - 1), mean


def transform(X: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Project data onto the principal directions: coordinates in the new basis."""
    return (np.asarray(X, dtype=float) - mean) @ components.T


def inverse_transform(Z: np.ndarray, components: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Map low-dimensional coordinates back to the original space (lossy if k < d)."""
    return Z @ components + mean


def explained_variance_ratio(X: np.ndarray, explained_variance: np.ndarray) -> np.ndarray:
    """Fraction of the total variance captured by each kept component."""
    total = np.asarray(X, dtype=float).var(axis=0, ddof=1).sum()
    return explained_variance / total
