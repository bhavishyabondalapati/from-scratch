"""Support vector machines.

LinearSVM  - primal problem solved by subgradient descent on the hinge loss.
KernelSVM  - dual problem solved by SMO (sequential minimal optimisation),
             which allows the kernel trick (RBF, polynomial, ...).
Both are binary classifiers; labels may be any two values.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# --- Kernels: K(a, b) = inner product of a and b in some feature space ---------


def linear_kernel(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B.T


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """exp(-gamma * ||a - b||^2): similarity that decays with distance.
    Corresponds to an infinite-dimensional feature space."""
    sq = (A**2).sum(1)[:, None] - 2 * A @ B.T + (B**2).sum(1)[None, :]
    return np.exp(-gamma * np.maximum(sq, 0))


def polynomial_kernel(
    A: np.ndarray, B: np.ndarray, degree: int = 3, gamma: float = 1.0, coef0: float = 1.0
) -> np.ndarray:
    """(gamma * a.b + coef0)^degree: all feature products up to `degree`, computed implicitly."""
    return (gamma * A @ B.T + coef0) ** degree


def _to_pm1(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, idx = np.unique(y, return_inverse=True)
    if len(classes) != 2:
        raise ValueError("binary classification only")
    return classes, np.where(idx == 1, 1.0, -1.0)


class LinearSVM:
    """Minimise  (lam / 2) ||w||^2 + mean(max(0, 1 - y (w.x + b))).

    The hinge loss is zero for points classified correctly with margin >= 1 and
    grows linearly otherwise. It has a kink, so we use a SUBgradient:
    -y*x for points inside the margin, 0 for the rest. Step size 1 / (lam * t)
    is the Pegasos schedule.
    """

    def __init__(self, lam: float = 0.01, n_iters: int = 2000) -> None:
        self.lam, self.n_iters = lam, n_iters

    def objective(self, X: np.ndarray, y_pm: np.ndarray, w: np.ndarray, b: float) -> float:
        hinge = np.maximum(0, 1 - y_pm * (X @ w + b))
        return float(0.5 * self.lam * w @ w + hinge.mean())

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearSVM:
        X = np.asarray(X, dtype=float)
        self.classes_, y_pm = _to_pm1(y)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        best = (np.inf, w.copy(), b)
        for t in range(1, self.n_iters + 1):
            lr = 1 / (self.lam * t)
            violators = y_pm * (X @ w + b) < 1  # points inside the margin or misclassified
            grad_w = self.lam * w - (y_pm[violators, None] * X[violators]).sum(axis=0) / n
            grad_b = -y_pm[violators].sum() / n
            w = w - lr * grad_w
            b = b - lr * grad_b
            # Subgradient descent is not monotone, so remember the best iterate seen.
            obj = self.objective(X, y_pm, w, b)
            if obj < best[0]:
                best = (obj, w.copy(), b)
        self.objective_, self.coef_, self.intercept_ = best
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]


class KernelSVM:
    """Soft-margin SVM trained in the dual with SMO.

    Dual problem (one variable alpha_i per training point):
        minimise   0.5 * sum_ij alpha_i alpha_j y_i y_j K(x_i, x_j) - sum_i alpha_i
        subject to 0 <= alpha_i <= C   and   sum_i alpha_i y_i = 0

    SMO: because of the equality constraint we cannot change one alpha alone,
    so repeatedly pick the PAIR that violates the optimality conditions the
    most and solve for those two analytically (a 1D quadratic, then clip to
    the box). Points that end with alpha > 0 are the support vectors.
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str | Callable[[np.ndarray, np.ndarray], np.ndarray] = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        tol: float = 1e-5,
        max_iter: int = 200_000,
    ) -> None:
        self.C, self.kernel, self.gamma, self.degree, self.coef0 = C, kernel, gamma, degree, coef0
        self.tol, self.max_iter = tol, max_iter

    def _K(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if callable(self.kernel):
            return self.kernel(A, B)
        if self.kernel == "linear":
            return linear_kernel(A, B)
        if self.kernel == "rbf":
            return rbf_kernel(A, B, self.gamma)
        if self.kernel == "poly":
            return polynomial_kernel(A, B, self.degree, self.gamma, self.coef0)
        raise ValueError(f"unknown kernel {self.kernel!r}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> KernelSVM:
        X = np.asarray(X, dtype=float)
        self.classes_, y_pm = _to_pm1(y)
        n, C = len(X), self.C
        K = self._K(X, X)
        alpha = np.zeros(n)
        grad = -np.ones(n)  # gradient of the dual objective; at alpha = 0 it is -1
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            # -y_i * grad_i is how much the objective improves by pushing alpha_i "up".
            score = -y_pm * grad
            can_go_up = ((y_pm > 0) & (alpha < C)) | ((y_pm < 0) & (alpha > 0))
            can_go_down = ((y_pm > 0) & (alpha > 0)) | ((y_pm < 0) & (alpha < C))
            i = int(np.argmax(np.where(can_go_up, score, -np.inf)))
            j = int(np.argmin(np.where(can_go_down, score, np.inf)))
            gap = score[i] - score[j]
            if gap < self.tol:  # KKT conditions hold: optimal
                break
            self.n_iter_ += 1
            # Move along alpha_i += y_i * t, alpha_j -= y_j * t (keeps sum alpha*y fixed).
            curvature = max(K[i, i] + K[j, j] - 2 * K[i, j], 1e-12)
            t = gap / curvature
            room_i = C - alpha[i] if y_pm[i] > 0 else alpha[i]
            room_j = alpha[j] if y_pm[j] > 0 else C - alpha[j]
            t = min(t, room_i, room_j)  # stay inside the box [0, C]
            alpha[i] += y_pm[i] * t
            alpha[j] -= y_pm[j] * t
            grad += y_pm * t * (K[:, i] - K[:, j])
        # The bias sits midway between the last most-violating pair's scores.
        self.intercept_ = float((score[i] + score[j]) / 2)
        support = alpha > 1e-8
        self.support_ = np.flatnonzero(support)
        self.support_vectors_ = X[support]
        self.dual_coef_ = (alpha * y_pm)[support]  # alpha_i * y_i
        self.alpha_ = alpha
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """f(x) = sum_i alpha_i y_i K(x_i, x) + b, using only the support vectors."""
        K = self._K(np.asarray(X, dtype=float), self.support_vectors_)
        return K @ self.dual_coef_ + self.intercept_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]
