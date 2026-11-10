"""Batch normalisation and layer normalisation with hand-written backward passes."""

from __future__ import annotations

import numpy as np

from s4_deep_learning.mlp.impl import Layer


def _normalise_backward(grad_xhat: np.ndarray, xhat: np.ndarray, inv_std: np.ndarray, axis) -> np.ndarray:
    """Backward through xhat = (x - mean) / sqrt(var + eps), where mean and var are
    taken over `axis`. Because every x affects the mean and variance too, the
    gradient has three parts; simplified, it is

        dx = inv_std * (g - mean(g) - xhat * mean(g * xhat))      (means over `axis`)
    """
    mean_g = grad_xhat.mean(axis=axis, keepdims=True)
    mean_gx = (grad_xhat * xhat).mean(axis=axis, keepdims=True)
    return inv_std * (grad_xhat - mean_g - xhat * mean_gx)


class BatchNorm1d(Layer):
    """Normalise every FEATURE using statistics over the BATCH, then scale and shift.

    Training: use this batch's mean/variance, and keep exponential running
    averages. Evaluation: use the running averages (a single test example has
    no batch to compute statistics from).
    """

    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        self.params = {"gamma": np.ones(n_features), "beta": np.zeros(n_features)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.momentum, self.eps = momentum, eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            mean, var = x.mean(axis=0), x.var(axis=0)
            n = len(x)
            # PyTorch convention: running variance uses the unbiased (n - 1) estimate.
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var * n / max(n - 1, 1)
        else:
            mean, var = self.running_mean, self.running_var
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mean) * self.inv_std
        return self.params["gamma"] * self.xhat + self.params["beta"]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        self.grads["gamma"] = (grad_out * self.xhat).sum(axis=0)
        self.grads["beta"] = grad_out.sum(axis=0)
        grad_xhat = grad_out * self.params["gamma"]
        if not self.training:  # statistics are constants at eval time
            return grad_xhat * self.inv_std
        return _normalise_backward(grad_xhat, self.xhat, self.inv_std, axis=0)


class LayerNorm(Layer):
    """Normalise every SAMPLE across its features. Same computation at train and
    test time and independent of batch size, which is why transformers use it."""

    def __init__(self, n_features: int, eps: float = 1e-5) -> None:
        self.params = {"gamma": np.ones(n_features), "beta": np.zeros(n_features)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.inv_std = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mean) * self.inv_std
        return self.params["gamma"] * self.xhat + self.params["beta"]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        batch_axes = tuple(range(grad_out.ndim - 1))
        self.grads["gamma"] = (grad_out * self.xhat).sum(axis=batch_axes)
        self.grads["beta"] = grad_out.sum(axis=batch_axes)
        return _normalise_backward(grad_out * self.params["gamma"], self.xhat, self.inv_std, axis=-1)
