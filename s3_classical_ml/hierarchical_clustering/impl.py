"""Agglomerative (bottom-up) hierarchical clustering with single/complete/average/ward linkage."""

from __future__ import annotations

import numpy as np

from s3_classical_ml.knn.impl import pairwise_sq_distances


def _lance_williams(
    linkage: str,
    d_ik: np.ndarray,
    d_jk: np.ndarray,
    d_ij: float,
    n_i: int,
    n_j: int,
    n_k: np.ndarray,
) -> np.ndarray:
    """Distance from the newly merged cluster (i + j) to every other cluster k,
    computed from the old distances only (the Lance-Williams update formula)."""
    if linkage == "single":  # closest pair of points
        return np.minimum(d_ik, d_jk)
    if linkage == "complete":  # farthest pair of points
        return np.maximum(d_ik, d_jk)
    if linkage == "average":  # mean over all point pairs
        return (n_i * d_ik + n_j * d_jk) / (n_i + n_j)
    if linkage == "ward":  # increase in within-cluster variance (on Euclidean distances)
        total = n_i + n_j + n_k
        sq = ((n_i + n_k) * d_ik**2 + (n_j + n_k) * d_jk**2 - n_k * d_ij**2) / total
        return np.sqrt(np.maximum(sq, 0))
    raise ValueError(f"unknown linkage {linkage!r}")


def linkage_matrix(X: np.ndarray, method: str = "average") -> np.ndarray:
    """Build the full merge tree. Returns a SciPy-style (n-1, 4) linkage matrix:
    each row is [cluster a, cluster b, merge distance, size of the new cluster].
    Original points are clusters 0..n-1; the merge in row t creates cluster n + t.

    Naive O(n^3): repeatedly find the closest pair of active clusters, merge
    them, and update one row/column of the distance matrix.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    D = np.sqrt(pairwise_sq_distances(X, X))
    np.fill_diagonal(D, np.inf)
    size = np.ones(n, dtype=int)
    cluster_id = np.arange(n)  # label of the cluster currently stored in each row
    active = np.ones(n, dtype=bool)
    Z = np.zeros((n - 1, 4))
    for t in range(n - 1):
        masked = np.where(active[:, None] & active[None, :], D, np.inf)
        i, j = np.unravel_index(np.argmin(masked), D.shape)
        i, j = min(i, j), max(i, j)
        d_ij = D[i, j]
        Z[t] = [
            min(cluster_id[i], cluster_id[j]),
            max(cluster_id[i], cluster_id[j]),
            d_ij,
            size[i] + size[j],
        ]
        new_row = _lance_williams(method, D[i], D[j], d_ij, size[i], size[j], size)
        D[i, :] = D[:, i] = new_row  # row i now holds the merged cluster
        D[i, i] = np.inf
        active[j] = False
        size[i] += size[j]
        cluster_id[i] = n + t
    return Z


def cut_tree(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    """Flat labels obtained by undoing the last (n_clusters - 1) merges."""
    n = len(Z) + 1
    parent = list(range(2 * n - 1))

    def find(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for t in range(n - n_clusters):  # replay only the first merges
        a, b = int(Z[t, 0]), int(Z[t, 1])
        parent[a] = parent[b] = n + t
    roots = [find(i) for i in range(n)]
    _, labels = np.unique(roots, return_inverse=True)
    return labels


class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 2, linkage: str = "ward") -> None:
        self.n_clusters, self.linkage = n_clusters, linkage

    def fit(self, X: np.ndarray) -> AgglomerativeClustering:
        self.linkage_matrix_ = linkage_matrix(X, self.linkage)
        self.labels_ = cut_tree(self.linkage_matrix_, self.n_clusters)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
