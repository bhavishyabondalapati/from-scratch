"""DBSCAN: density-based clustering that finds arbitrary shapes and marks noise."""

from __future__ import annotations

from collections import deque

import numpy as np

from s3_classical_ml.knn.impl import pairwise_sq_distances

NOISE = -1


class DBSCAN:
    """A point is a CORE point if at least `min_samples` points (itself included)
    lie within distance `eps`. Clusters grow outward from core points: every
    neighbour of a core point joins its cluster, and the growth continues only
    through neighbours that are core points themselves. Points reachable from
    no core point are noise (label -1).
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        self.eps, self.min_samples = eps, min_samples

    def fit(self, X: np.ndarray) -> DBSCAN:
        X = np.asarray(X, dtype=float)
        n = len(X)
        within = pairwise_sq_distances(X, X) <= self.eps**2  # O(n^2) neighbour table
        neighbours = [np.flatnonzero(row) for row in within]
        is_core = np.array([len(nb) >= self.min_samples for nb in neighbours])
        labels = np.full(n, NOISE)
        cluster = 0
        for start in range(n):
            if labels[start] != NOISE or not is_core[start]:
                continue
            # Breadth-first expansion from an unvisited core point.
            labels[start] = cluster
            queue = deque([start])
            while queue:
                p = queue.popleft()
                if not is_core[p]:
                    continue  # border point: belongs to the cluster but does not extend it
                for q in neighbours[p]:
                    if labels[q] == NOISE:
                        labels[q] = cluster
                        queue.append(q)
            cluster += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.flatnonzero(is_core)
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
