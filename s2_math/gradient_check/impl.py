"""Numerical gradient checking with finite differences.

This module is reused by later sections: every hand-written backward pass in
the repo is tested against `numerical_gradient`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def numerical_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Gradient of a scalar function f at x by central differences.

        df/dx_i ~= (f(x + eps*e_i) - f(x - eps*e_i)) / (2 * eps)

    The central difference has error O(eps^2); the one-sided version
    (f(x + eps) - f(x)) / eps only O(eps). x is perturbed in place and
    restored, so f may close over x (e.g. x is a layer's weight array).
    """
    if not np.issubdtype(x.dtype, np.floating):
        raise TypeError("x must be a float array (an int array cannot be nudged by eps)")
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        f_plus = f(x)
        x[idx] = original - eps
        f_minus = f(x)
        x[idx] = original  # restore
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def forward_difference_gradient(
    f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """One-sided differences: cheaper (one extra call per coordinate) but less accurate."""
    grad = np.zeros_like(x, dtype=float)
    f0 = f(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        original = x[idx]
        x[idx] = original + eps
        grad[idx] = (f(x) - f0) / eps
        x[idx] = original
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max(|a| + |b|, tiny). Scale-free, so 1e-7 means 'matches'
    whether the gradients are of size 1e-3 or 1e3."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-12)))


def check_gradient(
    f: Callable[[np.ndarray], float],
    analytic_grad: np.ndarray,
    x: np.ndarray,
    eps: float = 1e-5,
    tol: float = 1e-6,
) -> tuple[bool, float]:
    """Compare an analytic gradient to the numerical one. Returns (ok, relative error)."""
    err = relative_error(analytic_grad, numerical_gradient(f, x, eps))
    return err < tol, err


def numerical_jacobian(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Jacobian of a vector-valued f: J[i, j] = d f_i / d x_j (inputs/outputs flattened)."""
    x = np.asarray(x, dtype=float)
    out_size = np.asarray(f(x)).size
    J = np.zeros((out_size, x.size))
    flat = x.reshape(-1)  # a view: writing to flat writes to x
    for j in range(x.size):
        original = flat[j]
        flat[j] = original + eps
        f_plus = np.asarray(f(x), dtype=float).reshape(-1).copy()
        flat[j] = original - eps
        f_minus = np.asarray(f(x), dtype=float).reshape(-1).copy()
        flat[j] = original
        J[:, j] = (f_plus - f_minus) / (2 * eps)
    return J
