"""Logistic regression: binary (sigmoid) and multinomial (softmax), trained by gradient descent."""

from __future__ import annotations

import numpy as np

from s2_math.softmax_logsumexp.impl import (
    binary_cross_entropy_with_logits,
    cross_entropy,
    sigmoid,
    softmax,
)


def binary_loss_and_grad(
    w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray, l2: float = 0.0
) -> tuple[float, np.ndarray, float]:
    """Mean log-loss + (l2 / 2) * ||w||^2 and its gradient.

    The gradient has the same shape as in linear regression:
        dL/dw = X^T (p - y) / n + l2 * w,    dL/db = mean(p - y)
    where p = sigmoid(Xw + b) are the predicted probabilities.
    """
    z = X @ w + b
    loss = binary_cross_entropy_with_logits(z, y) + 0.5 * l2 * float(w @ w)
    err = sigmoid(z) - y
    return loss, X.T @ err / len(y) + l2 * w, float(err.mean())


class LogisticRegression:
    """Binary classifier: P(y = 1 | x) = sigmoid(w . x + b)."""

    def __init__(self, lr: float = 0.5, n_iters: int = 5000, l2: float = 0.0) -> None:
        self.lr, self.n_iters, self.l2 = lr, n_iters, l2
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LogisticRegression:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        w, b = np.zeros(X.shape[1]), 0.0
        for _ in range(self.n_iters):
            loss, grad_w, grad_b = binary_loss_and_grad(w, b, X, y, self.l2)
            self.loss_history_.append(loss)
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.coef_, self.intercept_ = w, b
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """The logit (log-odds): positive means class 1 is more likely."""
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def softmax_loss_and_grad(
    W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray, l2: float = 0.0
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean cross-entropy + (l2 / 2) * ||W||^2 for W of shape (features, classes).

    dL/dW = X^T (P - Y) / n + l2 * W, with P = softmax(XW + b) and Y one-hot.
    """
    logits = X @ W + b
    loss = cross_entropy(logits, y) + 0.5 * l2 * float(np.sum(W * W))
    err = softmax(logits)
    err[np.arange(len(y)), y] -= 1  # P - Y
    return loss, X.T @ err / len(y) + l2 * W, err.mean(axis=0)


class SoftmaxRegression:
    """Multinomial logistic regression: one weight vector per class, softmax on top."""

    def __init__(self, lr: float = 0.5, n_iters: int = 5000, l2: float = 0.0) -> None:
        self.lr, self.n_iters, self.l2 = lr, n_iters, l2
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> SoftmaxRegression:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        n_classes = int(y.max()) + 1
        W, b = np.zeros((X.shape[1], n_classes)), np.zeros(n_classes)
        for _ in range(self.n_iters):
            loss, grad_W, grad_b = softmax_loss_and_grad(W, b, X, y, self.l2)
            self.loss_history_.append(loss)
            W -= self.lr * grad_W
            b -= self.lr * grad_b
        self.coef_, self.intercept_ = W, b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return softmax(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
