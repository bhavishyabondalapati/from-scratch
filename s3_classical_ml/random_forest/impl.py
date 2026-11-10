"""Random forest = bagging (bootstrap samples) + random feature subsets at each split."""

from __future__ import annotations

import numpy as np

from s3_classical_ml.decision_tree.impl import DecisionTreeClassifier, DecisionTreeRegressor


class _BaseForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        max_features: int | str | None = "sqrt",
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state

    def _make_tree(self, seed: int):
        raise NotImplementedError

    def _fit_trees(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self.trees_ = []
        self.in_bag_ = []  # which rows each tree was trained on
        for _ in range(self.n_estimators):
            # Bootstrap: draw n rows WITH replacement. Each tree sees ~63% of the
            # distinct rows; the other ~37% are "out of bag" for that tree.
            idx = rng.integers(0, n, size=n) if self.bootstrap else np.arange(n)
            tree = self._make_tree(int(rng.integers(0, 2**31)))
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            self.in_bag_.append(idx)


class RandomForestClassifier(_BaseForest):
    def _make_tree(self, seed: int) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._fit_trees(X, y_idx)
        self.oob_score_ = self._oob_score(X, y_idx) if self.bootstrap else None
        return self

    def _tree_proba(self, tree: DecisionTreeClassifier, X: np.ndarray) -> np.ndarray:
        """A tree only knows the classes present in its bootstrap sample; spread
        its probabilities into columns for the forest's full class list."""
        out = np.zeros((len(X), len(self.classes_)))
        out[:, tree.classes_] = tree.predict_proba(X)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average the trees' class probabilities ("soft voting")."""
        X = np.asarray(X, dtype=float)
        return np.mean([self._tree_proba(t, X) for t in self.trees_], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def _oob_score(self, X: np.ndarray, y_idx: np.ndarray) -> float:
        """Accuracy where each row is predicted ONLY by trees that never saw it.
        A free validation estimate: no separate hold-out set needed."""
        votes = np.zeros((len(X), len(self.classes_)))
        for tree, idx in zip(self.trees_, self.in_bag_):
            oob = np.setdiff1d(np.arange(len(X)), idx)
            if len(oob):
                votes[oob] += self._tree_proba(tree, X[oob])
        scored = votes.sum(axis=1) > 0  # rows that were out-of-bag for at least one tree
        return float(np.mean(votes[scored].argmax(axis=1) == y_idx[scored]))


class RandomForestRegressor(_BaseForest):
    def __init__(self, max_features: int | str | None = None, **kwargs) -> None:
        super().__init__(max_features=max_features, **kwargs)  # regression default: all features

    def _make_tree(self, seed: int) -> DecisionTreeRegressor:
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        self._fit_trees(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.mean([t.predict(X) for t in self.trees_], axis=0)
