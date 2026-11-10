"""Numerically stable softmax, log-sum-exp, sigmoid and cross-entropy.

The naive versions are included on purpose so tests (and you) can see them break.
"""

from __future__ import annotations

import numpy as np


def naive_softmax(z: np.ndarray) -> np.ndarray:
    """Textbook formula. exp(1000) overflows to inf, giving inf / inf = nan."""
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def logsumexp(z: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """log(sum(exp(z))) without overflow or underflow.

    Factor out the maximum m:  log sum exp(z_i) = m + log sum exp(z_i - m).
    Now the largest exponent is exp(0) = 1, so nothing can overflow, and at
    least one term equals 1, so the sum cannot underflow to 0.
    """
    z = np.asarray(z, dtype=float)
    m = np.max(z, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)  # all -inf row: avoid (-inf) - (-inf) = nan
    out = m + np.log(np.sum(np.exp(z - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """exp(z_i) / sum_j exp(z_j), computed after subtracting the max.

    Softmax is unchanged by adding a constant to every logit, so shifting by
    the max changes nothing mathematically but keeps every exponent <= 0.
    """
    z = np.asarray(z, dtype=float)
    e = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """log(softmax(z)) = z - logsumexp(z).

    Computing log(softmax(z)) in two steps gives log(0) = -inf for very
    unlikely classes; this form stays finite.
    """
    z = np.asarray(z, dtype=float)
    return z - logsumexp(z, axis=axis, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean negative log-likelihood of integer class targets, from raw logits.

    logits: (n, classes), targets: (n,) ints. Works directly on logits (via
    log_softmax) instead of on probabilities, which is why frameworks ask you
    NOT to apply softmax before the loss.
    """
    logp = log_softmax(logits)
    return float(-np.mean(logp[np.arange(len(targets)), targets]))


def cross_entropy_grad(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """d(mean cross-entropy)/d(logits) = (softmax(logits) - one_hot(targets)) / n."""
    grad = softmax(logits)
    grad[np.arange(len(targets)), targets] -= 1
    return grad / len(targets)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """1 / (1 + exp(-x)) without overflow.

    For x >= 0 use 1 / (1 + exp(-x)); for x < 0 use exp(x) / (1 + exp(x)).
    Either way the exponent passed to exp is <= 0.
    """
    x = np.asarray(x, dtype=float)
    e = np.exp(-np.abs(x))  # always in (0, 1]
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    """log(sigmoid(x)) = -softplus(-x) = min(x, 0) - log(1 + exp(-|x|))."""
    x = np.asarray(x, dtype=float)
    return np.minimum(x, 0) - np.log1p(np.exp(-np.abs(x)))


def binary_cross_entropy_with_logits(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean of -[y log s(x) + (1 - y) log(1 - s(x))], computed from logits.

    Simplifies to max(x, 0) - x*y + log(1 + exp(-|x|)), which never takes log(0).
    """
    x, y = np.asarray(logits, dtype=float), np.asarray(targets, dtype=float)
    return float(np.mean(np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))))
