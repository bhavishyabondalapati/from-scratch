"""Naive Bayes classifiers: Gaussian (continuous features) and multinomial (counts)."""

from __future__ import annotations

import numpy as np

from s2_math.softmax_logsumexp.impl import logsumexp


class _NaiveBayesBase:
    """Shared prediction logic. Subclasses supply the joint log-likelihood
    log P(class) + sum_j log P(x_j | class), one column per class."""

    classes_: np.ndarray

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_likelihood(np.asarray(X, dtype=float))
        # Bayes' rule in log space: subtract log P(x) = logsumexp over classes.
        return joint - logsumexp(joint, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        joint = self._joint_log_likelihood(np.asarray(X, dtype=float))
        return self.classes_[joint.argmax(axis=1)]


class GaussianNB(_NaiveBayesBase):
    """Each feature, within each class, is modelled as an independent normal distribution.

    Fitting = computing a mean and a variance per (class, feature) pair.
    """

    def __init__(self, var_smoothing: float = 1e-9) -> None:
        self.var_smoothing = var_smoothing

    def fit(self, X: np.ndarray, y: np.ndarray) -> GaussianNB:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        # A small floor on the variance avoids dividing by zero for constant features.
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_])
        self.var_ += self.var_smoothing * X.var(axis=0).max()
        self.class_prior_ = np.array([np.mean(y == c) for c in self.classes_])
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        # log N(x; mu, var) = -0.5 * log(2 pi var) - (x - mu)^2 / (2 var), summed over features
        diff = X[:, None, :] - self.theta_[None, :, :]  # (n, classes, features)
        log_pdf = -0.5 * np.log(2 * np.pi * self.var_)[None] - diff**2 / (2 * self.var_[None])
        return np.log(self.class_prior_)[None, :] + log_pdf.sum(axis=2)


class MultinomialNB(_NaiveBayesBase):
    """For count features (e.g. word counts). Each class has its own distribution
    over features; a document is a bag of independent draws from it.

    alpha is Laplace smoothing: pretend every feature was seen `alpha` extra
    times in every class, so an unseen word does not zero out the whole product.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> MultinomialNB:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if (X < 0).any():
            raise ValueError("MultinomialNB needs non-negative counts")
        self.classes_ = np.unique(y)
        counts = np.array([X[y == c].sum(axis=0) for c in self.classes_]) + self.alpha
        self.feature_log_prob_ = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(np.array([np.mean(y == c) for c in self.classes_]))
        return self

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        # sum_j count_j * log P(feature j | class): a linear function of the counts!
        return X @ self.feature_log_prob_.T + self.class_log_prior_
