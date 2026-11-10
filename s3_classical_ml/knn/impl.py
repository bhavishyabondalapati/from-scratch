"""k-nearest neighbours for classification and regression (brute-force search)."""

from __future__ import annotations

import numpy as np


def pairwise_sq_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance between every row of A and every row of B.

    Uses ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, so the whole table is one
    matrix product instead of a double Python loop.
    """
    sq = (A**2).sum(axis=1)[:, None] - 2 * A @ B.T + (B**2).sum(axis=1)[None, :]
    return np.maximum(sq, 0)  # rounding can produce tiny negatives


def minkowski_distances(A: np.ndarray, B: np.ndarray, p: float = 2) -> np.ndarray:
    """General L_p distance table (p=1 Manhattan, p=2 Euclidean). O(n*m*d) memory."""
    diff = np.abs(A[:, None, :] - B[None, :, :])
    return (diff**p).sum(axis=2) ** (1 / p)


class _KNNBase:
    def __init__(self, k: int = 5, weights: str = "uniform") -> None:
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.k, self.weights = k, weights

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ "Training" is just remembering the data: KNN is a lazy learner."""
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y)
        return self

    def kneighbors(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(distances, indices) of the k nearest training points for each query row."""
        d = np.sqrt(pairwise_sq_distances(np.asarray(X, dtype=float), self.X_))
        # argpartition finds the k smallest in O(n) per row; then sort just those k.
        nearest = np.argpartition(d, self.k - 1, axis=1)[:, : self.k]
        rows = np.arange(len(d))[:, None]
        order = np.argsort(d[rows, nearest], axis=1, kind="stable")
        nearest = nearest[rows, order]
        return d[rows, nearest], nearest

    def _neighbor_weights(self, dist: np.ndarray) -> np.ndarray:
        if self.weights == "uniform":
            return np.ones_like(dist)
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        exact = np.isinf(w)  # query equals a training point: let exact matches decide alone
        rows_with_exact = exact.any(axis=1)
        w[rows_with_exact] = exact[rows_with_exact].astype(float)
        return w


class KNNClassifier(_KNNBase):
    """Predict the (optionally distance-weighted) majority class among the k nearest."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        super().fit(X, y)
        self.classes_, self._y_idx = np.unique(self.y_, return_inverse=True)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        dist, idx = self.kneighbors(X)
        w = self._neighbor_weights(dist)
        votes = np.zeros((len(dist), len(self.classes_)))
        rows = np.arange(len(dist))[:, None]
        np.add.at(votes, (rows, self._y_idx[idx]), w)  # add each neighbour's vote to its class
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # argmax returns the first maximum, so ties go to the smallest class label.
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class KNNRegressor(_KNNBase):
    """Predict the (optionally distance-weighted) mean target of the k nearest."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        dist, idx = self.kneighbors(X)
        w = self._neighbor_weights(dist)
        return (w * self.y_[idx]).sum(axis=1) / w.sum(axis=1)
