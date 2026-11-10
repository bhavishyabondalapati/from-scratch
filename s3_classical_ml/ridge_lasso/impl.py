"""Ridge (L2) and lasso (L1) regression.

Objectives follow scikit-learn's conventions so results can be compared directly:
    Ridge:  ||y - Xw - b||^2            + alpha * ||w||_2^2
    Lasso:  ||y - Xw - b||^2 / (2n)     + alpha * ||w||_1
The intercept b is never penalised (we centre X and y, fit w, then recover b).
"""

from __future__ import annotations

import numpy as np


class Ridge:
    """Least squares plus a penalty on the squared size of the weights.

    Closed form: w = (X^T X + alpha I)^-1 X^T y. Adding alpha to the diagonal
    makes the matrix invertible even with collinear features, and shrinks all
    weights smoothly toward zero (but never exactly to zero).
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> Ridge:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        d = X.shape[1]
        self.coef_ = np.linalg.solve(Xc.T @ Xc + self.alpha * np.eye(d), Xc.T @ yc)
        self.intercept_ = float(y_mean - x_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def soft_threshold(rho: float, alpha: float) -> float:
    """Shrink rho toward zero by alpha, and clamp to exactly zero inside [-alpha, alpha].

    This is the solution of the 1D lasso problem and the reason lasso
    produces exact zeros (sparse models).
    """
    if rho > alpha:
        return rho - alpha
    if rho < -alpha:
        return rho + alpha
    return 0.0


class Lasso:
    """Least squares plus a penalty on the absolute size of the weights.

    |w| has a kink at zero, so there is no closed form and plain gradient
    descent does not apply. Coordinate descent works beautifully instead:
    optimise ONE weight at a time with the others frozen. That 1D problem has
    the exact solution

        w_j = soft_threshold(rho_j, alpha) / z_j
        rho_j = (1/n) x_j . (residual with feature j's contribution added back)
        z_j   = (1/n) x_j . x_j
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-8) -> None:
        self.alpha, self.max_iter, self.tol = alpha, max_iter, tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> Lasso:
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n, d = X.shape
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
        w = np.zeros(d)
        residual = yc.copy()  # = yc - Xc @ w, kept up to date incrementally
        z = (Xc**2).sum(axis=0) / n
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            self.n_iter_ += 1
            biggest_change = 0.0
            for j in range(d):
                if z[j] == 0:
                    continue  # constant feature: carries no information
                old = w[j]
                rho = Xc[:, j] @ residual / n + z[j] * old  # correlation with partial residual
                w[j] = soft_threshold(rho, self.alpha) / z[j]
                if w[j] != old:
                    residual -= Xc[:, j] * (w[j] - old)
                    biggest_change = max(biggest_change, abs(w[j] - old))
            if biggest_change < self.tol:
                break
        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def lasso_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, alpha: float) -> float:
    return float(np.sum((y - X @ w - b) ** 2) / (2 * len(y)) + alpha * np.sum(np.abs(w)))


def ridge_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, alpha: float) -> float:
    return float(np.sum((y - X @ w - b) ** 2) + alpha * np.sum(w**2))
