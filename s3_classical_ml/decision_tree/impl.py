"""CART decision trees: classification (gini / entropy) and regression (variance).

Supports per-sample weights and random feature subsets, because random forests,
gradient boosting and AdaBoost in the neighbouring folders reuse these classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gini(counts: np.ndarray) -> np.ndarray:
    """Gini impurity 1 - sum_k p_k^2 from (weighted) class counts along the last axis.

    It is the chance of mislabelling a random sample if you label it by
    drawing from the node's own class distribution. 0 = pure node.
    """
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(total, 1e-300)
    return 1.0 - (p**2).sum(axis=-1)


def entropy(counts: np.ndarray) -> np.ndarray:
    """Entropy -sum_k p_k log2 p_k in bits. 0 = pure, log2(K) = uniform."""
    total = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(total, 1e-300)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)  # treat 0 * log 0 as 0
    return -(p * logp).sum(axis=-1)


def variance_from_stats(stats: np.ndarray) -> np.ndarray:
    """Variance from sufficient statistics [sum w, sum w*y, sum w*y^2] (last axis)."""
    w = np.maximum(stats[..., 0], 1e-300)
    mean = stats[..., 1] / w
    return np.maximum(stats[..., 2] / w - mean**2, 0.0)


@dataclass
class Node:
    """A leaf has `value` only; an internal node also has a split and two children."""

    value: np.ndarray | float  # class distribution (classifier) or mean target (regressor)
    n_samples: int
    impurity: float
    feature: int = -1
    threshold: float = 0.0
    left: Node | None = None  # samples with x[feature] <= threshold
    right: Node | None = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None


class _BaseTree:
    """Greedy top-down tree building shared by the classifier and the regressor.

    Each sample contributes a row of "sufficient statistics" (one-hot class for
    classification; [1, y, y^2] for regression). Sums of those rows are all we
    need to compute a node's impurity, so every possible split of a sorted
    feature can be scored at once with a cumulative sum.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: int | str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    # Subclasses define these three.
    def _stats(self, y: np.ndarray) -> np.ndarray: ...
    def _impurity(self, stats: np.ndarray) -> np.ndarray: ...
    def _leaf_value(self, stats: np.ndarray) -> np.ndarray | float: ...

    def _fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> None:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        if self.max_features is None:
            self._n_try = d
        elif self.max_features == "sqrt":
            self._n_try = max(1, int(np.sqrt(d)))
        else:
            self._n_try = int(self.max_features)
        self.n_features_ = d
        self.n_leaves_ = 0
        weighted_stats = self._stats(y) * w[:, None]
        self.root_ = self._grow(X, weighted_stats, depth=0)

    def _grow(self, X: np.ndarray, S: np.ndarray, depth: int) -> Node:
        total = S.sum(axis=0)
        node = Node(
            value=self._leaf_value(total), n_samples=len(X), impurity=float(self._impurity(total))
        )
        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(X) < self.min_samples_split
            or node.impurity <= 1e-12  # already pure
        )
        split = None if stop else self._best_split(X, S, total, node.impurity)
        if split is None:
            self.n_leaves_ += 1
            return node
        node.feature, node.threshold = split
        go_left = X[:, node.feature] <= node.threshold
        node.left = self._grow(X[go_left], S[go_left], depth + 1)
        node.right = self._grow(X[~go_left], S[~go_left], depth + 1)
        return node

    def _best_split(
        self, X: np.ndarray, S: np.ndarray, total: np.ndarray, parent_impurity: float
    ) -> tuple[int, float] | None:
        """Try every threshold of every candidate feature; return the one that
        lowers the weighted child impurity the most (None if nothing helps)."""
        n, d = X.shape
        features = np.arange(d) if self._n_try >= d else self._rng.choice(d, self._n_try, False)
        total_weight = self._weight(total)
        best_score, best = parent_impurity - 1e-12, None
        leaf = self.min_samples_leaf
        for f in features:
            order = np.argsort(X[:, f], kind="stable")
            x_sorted = X[order, f]
            left = np.cumsum(S[order], axis=0)[:-1]  # stats of the first i+1 samples
            right = total - left
            # A split between positions i and i+1 is only possible if the values differ,
            # and both sides must keep at least min_samples_leaf samples.
            valid = x_sorted[1:] > x_sorted[:-1]
            valid[: leaf - 1] = False
            if leaf > 1:
                valid[n - leaf :] = False
            if not valid.any():
                continue
            score = (
                self._weight(left) * self._impurity(left)
                + self._weight(right) * self._impurity(right)
            ) / total_weight
            score[~valid] = np.inf
            i = int(np.argmin(score))
            if score[i] < best_score:
                best_score = score[i]
                best = (int(f), float((x_sorted[i] + x_sorted[i + 1]) / 2))  # midpoint
        return best

    def _weight(self, stats: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _leaf_for(self, x: np.ndarray) -> Node:
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def apply(self, X: np.ndarray) -> list[Node]:
        """The leaf each sample lands in (boosting uses this to edit leaf values)."""
        return [self._leaf_for(x) for x in np.asarray(X, dtype=float)]

    def depth(self) -> int:
        def walk(node: Node) -> int:
            return 0 if node.is_leaf else 1 + max(walk(node.left), walk(node.right))

        return walk(self.root_)


class DecisionTreeClassifier(_BaseTree):
    def __init__(self, criterion: str = "gini", **kwargs) -> None:
        if criterion not in {"gini", "entropy"}:
            raise ValueError("criterion must be 'gini' or 'entropy'")
        super().__init__(**kwargs)
        self.criterion = criterion

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> DecisionTreeClassifier:
        self.classes_, self._y_idx = np.unique(y, return_inverse=True)
        self._fit(X, self._y_idx, sample_weight)
        return self

    def _stats(self, y: np.ndarray) -> np.ndarray:
        return np.eye(len(self.classes_))[y]  # one-hot rows; sums are class counts

    def _weight(self, stats: np.ndarray) -> np.ndarray:
        return stats.sum(axis=-1)

    def _impurity(self, stats: np.ndarray) -> np.ndarray:
        return gini(stats) if self.criterion == "gini" else entropy(stats)

    def _leaf_value(self, stats: np.ndarray) -> np.ndarray:
        return stats / stats.sum()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.array([leaf.value for leaf in self.apply(X)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class DecisionTreeRegressor(_BaseTree):
    """Splits minimise the weighted variance of the children (= squared error)."""

    def fit(
        self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> DecisionTreeRegressor:
        self._fit(X, np.asarray(y, dtype=float), sample_weight)
        return self

    def _stats(self, y: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones_like(y), y, y**2])

    def _weight(self, stats: np.ndarray) -> np.ndarray:
        return stats[..., 0]

    def _impurity(self, stats: np.ndarray) -> np.ndarray:
        return variance_from_stats(stats)

    def _leaf_value(self, stats: np.ndarray) -> float:
        return float(stats[1] / stats[0])  # weighted mean of the targets

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([leaf.value for leaf in self.apply(X)])
