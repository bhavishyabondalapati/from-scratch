"""Dropout, weight decay, and learning-rate schedules for the NumPy MLP."""

from __future__ import annotations

import math

import numpy as np

from s4_deep_learning.mlp.impl import Layer, Sequential, SoftmaxCrossEntropy

# --- Dropout --------------------------------------------------------------------------


class Dropout(Layer):
    """Inverted dropout: during training zero each unit with probability p and
    scale the survivors by 1 / (1 - p), so the EXPECTED activation is unchanged
    and nothing needs to change at test time (dropout is simply switched off).
    """

    def __init__(self, p: float = 0.5, seed: int = 0) -> None:
        if not 0 <= p < 1:
            raise ValueError("p must be in [0, 1)")
        self.p = p
        self.rng = np.random.default_rng(seed)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if not self.training or self.p == 0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p) / (1 - self.p)
        return x * self.mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out if self.mask is None else grad_out * self.mask


# --- Learning-rate schedules: step number -> learning rate ---------------------------------


def constant_lr(base: float):
    return lambda step: base


def step_decay(base: float, drop: float = 0.1, every: int = 30):
    """Multiply by `drop` every `every` steps (classic ResNet schedule)."""
    return lambda step: base * drop ** (step // every)


def exponential_decay(base: float, gamma: float = 0.99):
    return lambda step: base * gamma**step


def cosine_with_warmup(base: float, total: int, warmup: int = 0, min_lr: float = 0.0):
    """Linear warm-up from 0 to `base`, then cosine decay to `min_lr` (transformer default).

    Warm-up avoids huge, badly-directed updates while Adam's statistics and the
    weights are still random; cosine decay lets training settle into a minimum.
    """

    def lr(step: int) -> float:
        if step < warmup:
            return base * (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * progress))

    return lr


# --- Training loop with weight decay and a schedule -------------------------------------------


def train(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr_schedule,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    batch_size: int = 64,
    seed: int = 0,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Mini-batch SGD with momentum.

    Weight decay: add `weight_decay * W` to every weight gradient, i.e. minimise
    loss + (weight_decay / 2) * ||W||^2. Biases are not decayed (they do not
    control the model's sensitivity to inputs).
    """
    rng = np.random.default_rng(seed)
    loss_fn = SoftmaxCrossEntropy()
    velocity: dict[tuple[int, str], np.ndarray] = {}
    history: dict[str, list[float]] = {"train_loss": [], "val_acc": [], "lr": []}
    step = 0
    for _ in range(epochs):
        model.train()
        order, total = rng.permutation(len(X)), 0.0
        for start in range(0, len(X), batch_size):
            idx = order[start : start + batch_size]
            loss = loss_fn.forward(model(X[idx]), y[idx])
            model.backward(loss_fn.backward())
            lr = lr_schedule(step)
            for layer, name in model.parameters():
                g = layer.grads[name]
                if name == "W" and weight_decay:
                    g = g + weight_decay * layer.params[name]
                key = (id(layer), name)
                v = momentum * velocity.get(key, 0.0) + g
                velocity[key] = v
                layer.params[name] -= lr * v
            total += loss * len(idx)
            step += 1
        history["train_loss"].append(total / len(X))
        history["lr"].append(lr)
        model.eval()
        if X_val is not None:
            history["val_acc"].append(float(np.mean(model(X_val).argmax(1) == y_val)))
    return history


def weight_norm(model: Sequential) -> float:
    return float(
        math.sqrt(
            sum(np.sum(layer.params["W"] ** 2) for layer, n in model.parameters() if n == "W")
        )
    )
