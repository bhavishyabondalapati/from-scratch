"""k-means clustering (Lloyd's algorithm) with random or k-means++ initialisation."""

from __future__ import annotations

import numpy as np

from s3_classical_ml.knn.impl import pairwise_sq_distances


def kmeans_plus_plus(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick k starting centres that are spread out.

    First centre: a random point. Each next centre: a random point chosen with
    probability proportional to its SQUARED distance to the nearest centre so
    far. Far-away points are likely picks, so centres rarely start bunched up.
    """
    centres = [X[rng.integers(len(X))]]
    closest_sq = pairwise_sq_distances(X, np.array(centres))[:, 0]
    for _ in range(1, k):
        total = closest_sq.sum()
        if total == 0:  # fewer distinct points than k: just pick any
            idx = rng.integers(len(X))
        else:
            idx = rng.choice(len(X), p=closest_sq / total)
        centres.append(X[idx])
        closest_sq = np.minimum(closest_sq, pairwise_sq_distances(X, X[idx : idx + 1])[:, 0])
    return np.array(centres)


class KMeans:
    """Partition data into k groups so that the within-cluster sum of squared
    distances to the cluster mean ("inertia") is small.

    Lloyd's algorithm alternates two steps, each of which can only lower inertia:
    1. assign every point to its nearest centre;
    2. move every centre to the mean of the points assigned to it.
    It stops when assignments stop changing (a local minimum). Because the
    result depends on the start, we run `n_init` times and keep the best.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-8,
        random_state: int | None = None,
    ) -> None:
        if init not in {"k-means++", "random"}:
            raise ValueError("init must be 'k-means++' or 'random'")
        self.n_clusters, self.init, self.n_init = n_clusters, init, n_init
        self.max_iter, self.tol, self.random_state = max_iter, tol, random_state

    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        k = self.n_clusters
        if self.init == "k-means++":
            centres = kmeans_plus_plus(X, k, rng)
        else:
            centres = X[rng.choice(len(X), size=k, replace=False)].copy()
        history = []
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):  # noqa: B007 (used after the loop)
            d = pairwise_sq_distances(X, centres)
            labels = d.argmin(axis=1)
            history.append(float(d[np.arange(len(X)), labels].sum()))
            new = centres.copy()
            for c in range(k):
                members = X[labels == c]
                if len(members):  # an empty cluster keeps its old centre
                    new[c] = members.mean(axis=0)
            shift = float(((new - centres) ** 2).sum())
            centres = new
            if shift <= self.tol:
                break
        d = pairwise_sq_distances(X, centres)
        labels = d.argmin(axis=1)
        inertia = float(d[np.arange(len(X)), labels].sum())
        return centres, labels, inertia, n_iter, history

    def fit(self, X: np.ndarray) -> KMeans:
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            run = self._single_run(X, rng)
            if best is None or run[2] < best[2]:
                best = run
        (
            self.cluster_centers_,
            self.labels_,
            self.inertia_,
            self.n_iter_,
            self.inertia_history_,
        ) = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return pairwise_sq_distances(np.asarray(X, dtype=float), self.cluster_centers_).argmin(
            axis=1
        )

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
