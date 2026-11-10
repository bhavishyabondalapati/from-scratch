"""AdaBoost (SAMME, discrete) with decision stumps as weak learners."""

from __future__ import annotations

import numpy as np

from s3_classical_ml.decision_tree.impl import DecisionTreeClassifier


class AdaBoostClassifier:
    """Adaptive boosting.

    Keep a weight per training sample (start uniform). Each round:
    1. fit a weak learner (a depth-1 tree, a "stump") to the WEIGHTED data;
    2. measure its weighted error `err`;
    3. give it a say  alpha = lr * [ln((1 - err) / err) + ln(K - 1)]
       (accurate learners get a big say; K = number of classes);
    4. multiply the weights of the samples it got WRONG by exp(alpha) and
       renormalise, so the next learner concentrates on the hard cases.
    The final prediction is a weighted vote of all learners.

    For K = 2 this is the original AdaBoost up to a constant factor in alpha.
    """

    def __init__(
        self, n_estimators: int = 50, learning_rate: float = 1.0, max_depth: int = 1
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> AdaBoostClassifier:
        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, K = len(X), len(self.classes_)
        w = np.full(n, 1 / n)
        self.estimators_: list[DecisionTreeClassifier] = []
        self.estimator_weights_: list[float] = []
        self.estimator_errors_: list[float] = []
        for _ in range(self.n_estimators):
            stump = DecisionTreeClassifier(max_depth=self.max_depth).fit(X, y_idx, sample_weight=w)
            wrong = stump.predict(X) != y_idx
            err = float(w[wrong].sum() / w.sum())
            if err >= 1 - 1 / K:
                break  # no better than random guessing: stop
            self.estimators_.append(stump)
            self.estimator_errors_.append(err)
            if err <= 0:
                self.estimator_weights_.append(1.0)  # a perfect learner: nothing left to fix
                break
            alpha = self.learning_rate * (np.log((1 - err) / err) + np.log(K - 1))
            self.estimator_weights_.append(float(alpha))
            w = w * np.exp(alpha * wrong)
            w /= w.sum()
        return self

    def _votes(self, X: np.ndarray, n_estimators: int | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        votes = np.zeros((len(X), len(self.classes_)))
        rows = np.arange(len(X))
        pairs = list(zip(self.estimators_, self.estimator_weights_))[:n_estimators]
        for stump, alpha in pairs:
            votes[rows, stump.predict(X)] += alpha
        return votes

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self._votes(X).argmax(axis=1)]

    def staged_predict(self, X: np.ndarray):
        """Predictions using the first 1, 2, ... learners."""
        for m in range(1, len(self.estimators_) + 1):
            yield self.classes_[self._votes(X, m).argmax(axis=1)]
