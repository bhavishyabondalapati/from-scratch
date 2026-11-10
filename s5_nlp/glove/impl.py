"""GloVe: word vectors from a weighted least-squares fit to log co-occurrence counts (NumPy)."""

from __future__ import annotations

import numpy as np

from s5_nlp.word2vec.impl import Vocab


def cooccurrence_matrix(sentences: list[list[str]], vocab: Vocab, window: int = 5) -> np.ndarray:
    """X[i, j] = sum over occurrences of word j within `window` of word i of 1 / distance.
    Symmetric. Nearby words count more than distant ones."""
    V = len(vocab)
    X = np.zeros((V, V))
    for s in sentences:
        ids = vocab.encode(s)
        for i, wi in enumerate(ids):
            for j in range(max(0, i - window), i):
                X[wi, ids[j]] += 1.0 / (i - j)
                X[ids[j], wi] += 1.0 / (i - j)
    return X


def weighting(X: np.ndarray, x_max: float = 100.0, alpha: float = 0.75) -> np.ndarray:
    """f(x) = (x / x_max)^alpha below x_max, else 1; and 0 where x = 0.
    Rare pairs are noisy so they get little weight; very frequent pairs are capped."""
    return np.where(X > 0, np.minimum((X / x_max) ** alpha, 1.0), 0.0)


class GloVe:
    """Find vectors w_i, context vectors c_j and biases so that

        w_i . c_j + b_i + b'_j  ~=  log X_ij      for every NON-ZERO X_ij

    by minimising J = sum_ij f(X_ij) (w_i . c_j + b_i + b'_j - log X_ij)^2.
    The final embedding is w + c (the two tables are symmetric in theory).
    """

    def __init__(self, vocab_size: int, dim: int = 30, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-0.5, 0.5, (vocab_size, dim)) / dim
        self.C = rng.uniform(-0.5, 0.5, (vocab_size, dim)) / dim
        self.b = np.zeros(vocab_size)
        self.bc = np.zeros(vocab_size)

    def loss_and_grads(self, X: np.ndarray, F: np.ndarray):
        """Full-batch loss and gradients (the matrix is small, so no sampling is needed)."""
        logX = np.log(np.where(X > 0, X, 1.0))
        err = self.W @ self.C.T + self.b[:, None] + self.bc[None, :] - logX
        err = np.where(X > 0, err, 0.0)
        weighted = F * err  # (V, V)
        loss = float(np.sum(F * err**2))
        grads = {
            "W": 2 * weighted @ self.C,
            "C": 2 * weighted.T @ self.W,
            "b": 2 * weighted.sum(axis=1),
            "bc": 2 * weighted.sum(axis=0),
        }
        return loss, grads

    def fit(
        self, X: np.ndarray, epochs: int = 300, lr: float = 0.05, x_max: float = 100.0
    ) -> list[float]:
        """AdaGrad, as in the original paper: each parameter's step shrinks with the
        sum of its past squared gradients."""
        F = weighting(X, x_max)
        params = {"W": self.W, "C": self.C, "b": self.b, "bc": self.bc}
        hist_sq = {k: np.full_like(v, 1e-8) for k, v in params.items()}
        history = []
        for _ in range(epochs):
            loss, grads = self.loss_and_grads(X, F)
            history.append(loss)
            for k, g in grads.items():
                hist_sq[k] += g**2
                params[k] -= lr * g / np.sqrt(hist_sq[k])
        return history

    @property
    def embeddings(self) -> np.ndarray:
        return self.W + self.C


def ppmi_svd_embeddings(X: np.ndarray, dim: int) -> np.ndarray:
    """A classic count-based baseline: positive PMI matrix, truncated SVD."""
    total = X.sum()
    row, col = X.sum(axis=1, keepdims=True), X.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore"):
        pmi = np.log(X * total / (row * col))
    ppmi = np.where(np.isfinite(pmi), np.maximum(pmi, 0), 0)
    U, S, _ = np.linalg.svd(ppmi)
    return U[:, :dim] * np.sqrt(S[:dim])
