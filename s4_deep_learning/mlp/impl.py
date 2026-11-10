"""Multi-layer perceptron with hand-written backprop (NumPy), activations and initialisations.

Each layer has forward(x) and backward(grad_out) -> grad_in. Layers cache what
they need from the forward pass. Parameter gradients land in layer.grads, so an
optimiser can update layer.params in place. The tensor autograd engine could
do this for us; writing it by hand shows exactly what backprop computes.
"""

from __future__ import annotations

import math

import numpy as np

from s2_math.softmax_logsumexp.impl import log_softmax, softmax

# --- Initialisation -----------------------------------------------------------------


def xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Glorot/Xavier uniform: Var(w) = 2 / (fan_in + fan_out).

    Keeps the variance of activations AND of gradients roughly constant across
    layers for symmetric activations like tanh (which is ~linear near zero).
    """
    limit = math.sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Kaiming/He normal: Var(w) = 2 / fan_in.

    ReLU zeroes half of its inputs, halving the variance; the factor 2 makes up for it.
    """
    return rng.normal(0.0, math.sqrt(2 / fan_in), size=(fan_in, fan_out))


INITS = {"xavier": xavier_init, "he": he_init}

# --- Layers -----------------------------------------------------------------------------


class Layer:
    params: dict[str, np.ndarray] = {}
    grads: dict[str, np.ndarray] = {}
    training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Linear(Layer):
    """y = x W + b.   dW = x^T G,  db = sum_rows G,  dx = G W^T."""

    def __init__(
        self, n_in: int, n_out: int, init: str = "he", rng: np.random.Generator | None = None
    ) -> None:
        rng = rng or np.random.default_rng(0)
        self.params = {"W": INITS[init](n_in, n_out, rng), "b": np.zeros(n_out)}
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        self.grads["W"] = self.x.T @ grad_out
        self.grads["b"] = grad_out.sum(axis=0)  # the bias was broadcast over the batch
        return grad_out @ self.params["W"].T


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out):
        return grad_out * self.mask


class LeakyReLU(Layer):
    def __init__(self, slope: float = 0.01) -> None:
        self.slope = slope

    def forward(self, x):
        self.x = x
        return np.where(x > 0, x, self.slope * x)

    def backward(self, grad_out):
        return grad_out * np.where(self.x > 0, 1.0, self.slope)


class Tanh(Layer):
    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad_out):
        return grad_out * (1 - self.y**2)


class Sigmoid(Layer):
    def forward(self, x):
        e = np.exp(-np.abs(x))
        self.y = np.where(x >= 0, 1 / (1 + e), e / (1 + e))
        return self.y

    def backward(self, grad_out):
        return grad_out * self.y * (1 - self.y)


class GELU(Layer):
    """x * Phi(x) with the tanh approximation used by GPT/BERT."""

    C = math.sqrt(2 / math.pi)

    def forward(self, x):
        self.x = x
        self.t = np.tanh(self.C * (x + 0.044715 * x**3))
        return 0.5 * x * (1 + self.t)

    def backward(self, grad_out):
        x, t = self.x, self.t
        dt = (1 - t**2) * self.C * (1 + 3 * 0.044715 * x**2)
        return grad_out * (0.5 * (1 + t) + 0.5 * x * dt)


ACTIVATIONS = {
    "relu": ReLU,
    "leaky_relu": LeakyReLU,
    "tanh": Tanh,
    "sigmoid": Sigmoid,
    "gelu": GELU,
}


class SoftmaxCrossEntropy:
    """Loss on logits; backward gives (softmax - one_hot) / batch in one step."""

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        self.logits, self.targets = logits, targets
        return float(-log_softmax(logits)[np.arange(len(targets)), targets].mean())

    def backward(self) -> np.ndarray:
        grad = softmax(self.logits)
        grad[np.arange(len(self.targets)), self.targets] -= 1
        return grad / len(self.targets)


class Sequential:
    def __init__(self, *layers: Layer) -> None:
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    __call__ = forward

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):  # chain rule, from the loss back to the input
            grad = layer.backward(grad)
        return grad

    def parameters(self):
        """Yields (layer, name) pairs so optimisers can update params in place."""
        for layer in self.layers:
            for name in layer.params:
                yield layer, name

    def train(self, mode: bool = True) -> None:
        for layer in self.layers:
            layer.training = mode

    def eval(self) -> None:
        self.train(False)


def make_mlp(
    sizes: list[int], activation: str = "relu", init: str = "he", seed: int = 0
) -> Sequential:
    """Linear -> act -> Linear -> act -> ... -> Linear (no activation on the output)."""
    rng = np.random.default_rng(seed)
    layers: list[Layer] = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], init, rng))
        if i < len(sizes) - 2:
            layers.append(ACTIVATIONS[activation]())
    return Sequential(*layers)


class SGD:
    def __init__(self, model: Sequential, lr: float = 0.1, momentum: float = 0.0) -> None:
        self.model, self.lr, self.momentum = model, lr, momentum
        self.velocity: dict[tuple[int, str], np.ndarray] = {}

    def step(self) -> None:
        for layer, name in self.model.parameters():
            key = (id(layer), name)
            v = self.velocity.get(key, np.zeros_like(layer.params[name]))
            v = self.momentum * v + layer.grads[name]
            self.velocity[key] = v
            layer.params[name] -= self.lr * v


def train_classifier(
    model: Sequential,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.1,
    momentum: float = 0.9,
    seed: int = 0,
) -> list[float]:
    """Mini-batch SGD with momentum. Returns the mean training loss of each epoch."""
    rng = np.random.default_rng(seed)
    loss_fn, opt = SoftmaxCrossEntropy(), SGD(model, lr, momentum)
    history = []
    for _ in range(epochs):
        model.train()
        order, total = rng.permutation(len(X)), 0.0
        for start in range(0, len(X), batch_size):
            idx = order[start : start + batch_size]
            loss = loss_fn.forward(model(X[idx]), y[idx])
            model.backward(loss_fn.backward())
            opt.step()
            total += loss * len(idx)
        history.append(total / len(X))
    model.eval()
    return history


def activation_stats(
    sizes: list[int], activation: str, init: str, n: int = 1000, seed: int = 0
) -> list[float]:
    """Standard deviation of the activations after each hidden layer at initialisation."""
    model = make_mlp(sizes, activation, init, seed)
    x = np.random.default_rng(seed + 1).normal(size=(n, sizes[0]))
    stds = []
    for layer in model.layers:
        x = layer.forward(x)
        if not isinstance(layer, Linear):
            stds.append(float(x.std()))
    return stds
