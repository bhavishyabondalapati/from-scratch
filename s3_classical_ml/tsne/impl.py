"""Exact t-SNE (O(n^2)) for small datasets, in NumPy."""

from __future__ import annotations

import numpy as np

from s3_classical_ml.knn.impl import pairwise_sq_distances


def conditional_probabilities(
    sq_dist: np.ndarray, perplexity: float, tol: float = 1e-5, max_iter: int = 100
) -> np.ndarray:
    """P[i, j] = p(j | i): a Gaussian around point i, with its width sigma_i
    chosen by binary search so that the distribution's perplexity 2^H matches
    the target. Dense regions get narrow Gaussians, sparse regions wide ones,
    so every point gets roughly `perplexity` effective neighbours.
    """
    n = len(sq_dist)
    P = np.zeros((n, n))
    target_entropy = np.log(perplexity)  # natural-log entropy of the target
    for i in range(n):
        d = np.delete(sq_dist[i], i)
        lo, hi, beta = 0.0, np.inf, 1.0  # beta = 1 / (2 sigma^2)
        for _ in range(max_iter):
            logits = -d * beta
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            H = -np.sum(p * np.log(np.maximum(p, 1e-300)))
            if abs(H - target_entropy) < tol:
                break
            if H > target_entropy:  # too spread out: narrow the Gaussian
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        P[i, np.arange(n) != i] = p
    return P


def kl_divergence_and_grad(P: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """KL(P || Q) and its gradient with respect to the embedding Y.

    Q uses a Student-t (1 degree of freedom) kernel in the low-dimensional
    space: q_ij ∝ 1 / (1 + ||y_i - y_j||^2). Its heavy tails let moderately
    far points sit far apart, fixing the "crowding problem".
        dC/dy_i = 4 * sum_j (p_ij - q_ij) * (y_i - y_j) / (1 + ||y_i - y_j||^2)
    """
    num = 1.0 / (1.0 + pairwise_sq_distances(Y, Y))
    np.fill_diagonal(num, 0.0)
    Q = np.maximum(num / num.sum(), 1e-12)
    kl = float(np.sum(P * np.log(np.maximum(P, 1e-12) / Q)))
    W = (P - Q) * num  # (n, n)
    grad = 4.0 * (W.sum(axis=1)[:, None] * Y - W @ Y)
    return kl, grad


class TSNE:
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        exaggeration_iters: int = 250,
        random_state: int | None = None,
    ) -> None:
        self.n_components, self.perplexity, self.learning_rate = (
            n_components,
            perplexity,
            learning_rate,
        )
        self.n_iter, self.early_exaggeration = n_iter, early_exaggeration
        self.exaggeration_iters, self.random_state = exaggeration_iters, random_state

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = len(X)
        P = conditional_probabilities(pairwise_sq_distances(X, X), self.perplexity)
        P = (P + P.T) / (2 * n)  # symmetrise: joint probabilities summing to 1
        P = np.maximum(P, 1e-12)
        rng = np.random.default_rng(self.random_state)
        Y = rng.normal(scale=1e-4, size=(n, self.n_components))
        velocity = np.zeros_like(Y)
        gains = np.ones_like(Y)
        self.kl_history_ = []
        for it in range(self.n_iter):
            exaggerate = it < self.exaggeration_iters
            # Early exaggeration: inflate P so clusters form tight, well-separated groups first.
            kl, grad = kl_divergence_and_grad(
                P * (self.early_exaggeration if exaggerate else 1.0), Y
            )
            if not exaggerate:
                self.kl_history_.append(kl)
            momentum = 0.5 if exaggerate else 0.8
            # Per-coordinate adaptive gains (as in the reference implementation).
            same_sign = np.sign(grad) == np.sign(velocity)
            gains = np.where(same_sign, gains * 0.8, gains + 0.2).clip(min=0.01)
            velocity = momentum * velocity - self.learning_rate * gains * grad
            Y = Y + velocity
            Y -= Y.mean(axis=0)
        self.kl_divergence_ = kl_divergence_and_grad(P, Y)[0]
        self.embedding_ = Y
        return Y
