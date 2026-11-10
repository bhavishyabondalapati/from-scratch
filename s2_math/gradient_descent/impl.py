"""Gradient descent variants: batch GD, SGD, momentum, RMSProp, Adam.

Each optimizer is a small class with one method, step(params, grad) -> new params,
so the update rule is the only thing on the page.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

GradFn = Callable[[np.ndarray], np.ndarray]


class SGD:
    """Plain gradient descent: move a small step against the gradient.

        x <- x - lr * g

    The same rule is called "batch GD" when g is computed on all the data and
    "stochastic GD" when g comes from a random mini-batch.
    """

    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        return x - self.lr * g


class Momentum:
    """SGD with momentum: keep a running "velocity" of past gradients.

        v <- mu * v + g
        x <- x - lr * v

    Like a heavy ball rolling downhill: it speeds up along directions where
    gradients keep agreeing and averages out directions where they flip sign.
    """

    def __init__(self, lr: float = 0.01, mu: float = 0.9) -> None:
        self.lr, self.mu = lr, mu
        self.v: np.ndarray | None = None

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        self.v = g.copy() if self.v is None else self.mu * self.v + g
        return x - self.lr * self.v


class RMSProp:
    """Per-coordinate step sizes from a running average of squared gradients.

        s <- rho * s + (1 - rho) * g^2
        x <- x - lr * g / (sqrt(s) + eps)

    Coordinates with consistently large gradients get smaller steps and vice
    versa, which fixes badly scaled problems.
    """

    def __init__(self, lr: float = 0.01, rho: float = 0.99, eps: float = 1e-8) -> None:
        self.lr, self.rho, self.eps = lr, rho, eps
        self.s: np.ndarray | None = None

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.zeros_like(g)
        self.s = self.rho * self.s + (1 - self.rho) * g**2
        return x - self.lr * g / (np.sqrt(self.s) + self.eps)


class Adam:
    """Adam = momentum (first moment m) + RMSProp (second moment v) + bias correction.

        m <- b1 * m + (1 - b1) * g
        v <- b2 * v + (1 - b2) * g^2
        m_hat = m / (1 - b1^t),  v_hat = v / (1 - b2^t)
        x <- x - lr * m_hat / (sqrt(v_hat) + eps)

    m and v start at zero, so early on they are biased toward zero; dividing
    by (1 - b^t) undoes exactly that bias.
    """

    def __init__(
        self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8
    ) -> None:
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m: np.ndarray | None = None
        self.v: np.ndarray | None = None
        self.t = 0

    def step(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m, self.v = np.zeros_like(g), np.zeros_like(g)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def minimize(
    grad_fn: GradFn, x0: np.ndarray, optimizer, steps: int = 1000
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run an optimizer for a fixed number of steps; returns (final x, path taken)."""
    x = np.asarray(x0, dtype=float).copy()
    path = [x.copy()]
    for _ in range(steps):
        x = optimizer.step(x, grad_fn(x))
        path.append(x.copy())
    return x, path


def minibatch_sgd(
    grad_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x0: np.ndarray,
    n_samples: int,
    optimizer,
    batch_size: int = 32,
    epochs: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """Stochastic gradient descent over a dataset of n_samples points.

    grad_fn(x, batch_indices) returns the gradient estimated on that batch.
    Each epoch shuffles the data and visits every sample exactly once.
    batch_size = n_samples gives batch GD; batch_size = 1 gives "pure" SGD.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    for _ in range(epochs):
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            x = optimizer.step(x, grad_fn(x, order[start : start + batch_size]))
    return x
