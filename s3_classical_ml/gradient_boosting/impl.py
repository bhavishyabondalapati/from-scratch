"""Gradient boosting with regression trees: squared loss (regression) and log-loss (binary)."""

from __future__ import annotations

import numpy as np

from s2_math.softmax_logsumexp.impl import sigmoid
from s3_classical_ml.decision_tree.impl import DecisionTreeRegressor


class GradientBoostingRegressor:
    """Additive model F(x) = F0 + lr * sum_m tree_m(x), built one tree at a time.

    Each new tree is fitted to the current RESIDUALS y - F(x). For squared loss
    the residual is exactly the negative gradient of the loss with respect to
    the prediction, which is where the name "gradient boosting" comes from:
    every round is one step of gradient descent in function space.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.init_ = float(y.mean())  # the best constant prediction for squared loss
        F = np.full(len(y), self.init_)
        self.trees_: list[DecisionTreeRegressor] = []
        self.train_loss_: list[float] = []
        for _ in range(self.n_estimators):
            residual = y - F  # negative gradient of 0.5 * (y - F)^2
            rows = self._sample_rows(rng, len(y))
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X[rows], residual[rows])
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            self.train_loss_.append(float(np.mean((y - F) ** 2)))
        return self

    def _sample_rows(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """subsample < 1 gives "stochastic gradient boosting": each tree sees a random part."""
        if self.subsample >= 1.0:
            return np.arange(n)
        return rng.choice(n, size=max(1, int(self.subsample * n)), replace=False)

    def staged_predict(self, X: np.ndarray):
        """Yield predictions after 1, 2, ... trees (to pick the best number of rounds)."""
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
            yield F

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.init_ + self.learning_rate * sum(t.predict(X) for t in self.trees_)


class GradientBoostingClassifier:
    """Binary classification with log-loss. F(x) is a log-odds score; p = sigmoid(F).

    Each round:
    1. pseudo-residuals r = y - p (the negative gradient of log-loss w.r.t. F);
    2. fit a regression tree to r (the tree decides how to group the samples);
    3. replace each leaf's value with one Newton step for the samples in it:
           gamma = sum(r) / sum(p * (1 - p))
       because the mean residual is not the loss-minimising step for log-loss.
    """

    def __init__(
        self, n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 3
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> GradientBoostingClassifier:
        X = np.asarray(X, dtype=float)
        self.classes_, y01 = np.unique(y, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError("this implementation handles binary classification only")
        y01 = y01.astype(float)
        prior = y01.mean()
        self.init_ = float(np.log(prior / (1 - prior)))  # log-odds of the base rate
        F = np.full(len(y01), self.init_)
        self.trees_: list[DecisionTreeRegressor] = []
        self.train_loss_: list[float] = []
        for _ in range(self.n_estimators):
            p = sigmoid(F)
            residual = y01 - p
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(X, residual)
            # Newton update of every leaf, accumulated over the samples that land in it.
            leaves = tree.apply(X)
            numerator: dict[int, float] = {}
            denominator: dict[int, float] = {}
            for leaf, r, h in zip(leaves, residual, p * (1 - p)):
                numerator[id(leaf)] = numerator.get(id(leaf), 0.0) + r
                denominator[id(leaf)] = denominator.get(id(leaf), 0.0) + h
            for leaf in {id(lf): lf for lf in leaves}.values():
                leaf.value = numerator[id(leaf)] / max(denominator[id(leaf)], 1e-12)
            F += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)
            p = np.clip(sigmoid(F), 1e-15, 1 - 1e-15)
            self.train_loss_.append(float(-np.mean(y01 * np.log(p) + (1 - y01) * np.log(1 - p))))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.init_ + self.learning_rate * sum(t.predict(X) for t in self.trees_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
