"""Linear regression: closed form (normal equations) and gradient descent."""

from __future__ import annotations

import numpy as np


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Append a column of ones so the intercept is just another weight."""
    return np.hstack([X, np.ones((len(X), 1))])


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """1 - (model's squared error) / (squared error of always predicting the mean).

    1.0 = perfect, 0.0 = no better than the mean, negative = worse than the mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def mse_gradient(Xb: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Gradient of mean((Xb w - y)^2) with respect to w:  (2 / n) Xb^T (Xb w - y)."""
    return 2 * Xb.T @ (Xb @ w - y) / len(y)


class LinearRegression:
    """Ordinary least squares: find w, b minimising mean((X w + b - y)^2).

    method="closed_form": solve the normal equations (X^T X) w = X^T y exactly.
    method="gd":          start at zero and follow the negative gradient.
    """

    def __init__(self, method: str = "closed_form", lr: float = 0.05, n_iters: int = 5000) -> None:
        if method not in {"closed_form", "gd"}:
            raise ValueError("method must be 'closed_form' or 'gd'")
        self.method, self.lr, self.n_iters = method, lr, n_iters
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.loss_history_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        Xb = add_bias_column(X)
        if self.method == "closed_form":
            # lstsq solves the same least-squares problem as the normal equations
            # but via SVD, so it still works when X^T X is singular (collinear features).
            w = np.linalg.lstsq(Xb, y, rcond=None)[0]
        else:
            w = np.zeros(Xb.shape[1])
            for _ in range(self.n_iters):
                self.loss_history_.append(mse(y, Xb @ w))
                w -= self.lr * mse_gradient(Xb, y, w)
        self.coef_, self.intercept_ = w[:-1], float(w[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return r2_score(np.asarray(y, dtype=float), self.predict(X))


def normal_equations(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The textbook formula w = (X^T X)^-1 X^T y, written with solve() rather
    than an explicit inverse. X must already include a bias column if wanted."""
    return np.linalg.solve(X.T @ X, X.T @ y)
