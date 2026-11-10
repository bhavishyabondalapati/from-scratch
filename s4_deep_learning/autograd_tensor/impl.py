"""A tensor automatic-differentiation engine in NumPy.

Same idea as the scalar engine, but every node holds a whole array, so one
Python-level operation does millions of multiplications. The extra work is
broadcasting: if an input was broadcast up to a bigger shape in the forward
pass, its gradient must be SUMMED back down to its original shape.

Later folders in Section 4 (mlp, regularization, normalization) build on this.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum `grad` over the axes that broadcasting added or stretched, giving `shape`."""
    while grad.ndim > len(shape):  # leading axes that did not exist
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):  # axes that were size 1 and got stretched
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: tuple[Tensor, ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad: np.ndarray | None = None
        self._backward: Callable[[], None] = lambda: None
        self._prev = _children
        self._op = _op

    # --- helpers ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def _accumulate(self, g: np.ndarray) -> None:
        if not self.requires_grad:
            return  # constants (inputs, targets) never need a gradient
        g = _unbroadcast(g, self.shape)
        self.grad = g.copy() if self.grad is None else self.grad + g

    @staticmethod
    def _wrap(x) -> Tensor:
        return x if isinstance(x, Tensor) else Tensor(x)

    def _make(self, data: np.ndarray, children: tuple[Tensor, ...], op: str) -> Tensor:
        return Tensor(data, any(c.requires_grad for c in children), children, op)

    # --- elementwise arithmetic (with broadcasting) -----------------------------------

    def __add__(self, other) -> Tensor:
        other = self._wrap(other)
        out = self._make(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self._accumulate(out.grad)
            other._accumulate(out.grad)

        out._backward = _backward
        return out

    def __mul__(self, other) -> Tensor:
        other = self._wrap(other)
        out = self._make(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self._accumulate(out.grad * other.data)
            other._accumulate(out.grad * self.data)

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Tensor:
        out = self._make(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self._accumulate(out.grad * exponent * self.data ** (exponent - 1))

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1.0

    def __sub__(self, other) -> Tensor:
        return self + (-self._wrap(other))

    def __truediv__(self, other) -> Tensor:
        return self * self._wrap(other) ** -1.0

    def __radd__(self, other) -> Tensor:
        return self + other

    def __rsub__(self, other) -> Tensor:
        return self._wrap(other) - self

    def __rmul__(self, other) -> Tensor:
        return self * other

    def __rtruediv__(self, other) -> Tensor:
        return self._wrap(other) / self

    # --- matrix multiply ---------------------------------------------------------------

    def __matmul__(self, other) -> Tensor:
        """C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC  (for 2D; batched via swapaxes)."""
        other = self._wrap(other)
        out = self._make(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            g = out.grad
            a, b = self.data, other.data
            if a.ndim == 1:  # vector @ matrix
                self._accumulate(g @ b.T)
                other._accumulate(np.outer(a, g))
            elif b.ndim == 1:  # matrix @ vector
                self._accumulate(np.outer(g, b) if a.ndim == 2 else g[..., None] * b)
                other._accumulate(a.T @ g if a.ndim == 2 else np.einsum("...i,...ij->j", g, a))
            else:
                self._accumulate(g @ np.swapaxes(b, -1, -2))
                other._accumulate(np.swapaxes(a, -1, -2) @ g)

        out._backward = _backward
        return out

    # --- unary functions --------------------------------------------------------------

    def _unary(self, value: np.ndarray, local_grad: np.ndarray, op: str) -> Tensor:
        out = self._make(value, (self,), op)

        def _backward() -> None:
            self._accumulate(out.grad * local_grad)

        out._backward = _backward
        return out

    def exp(self) -> Tensor:
        e = np.exp(self.data)
        return self._unary(e, e, "exp")

    def log(self) -> Tensor:
        return self._unary(np.log(self.data), 1.0 / self.data, "log")

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        return self._unary(t, 1 - t**2, "tanh")

    def relu(self) -> Tensor:
        return self._unary(np.maximum(self.data, 0), (self.data > 0).astype(float), "relu")

    def sigmoid(self) -> Tensor:
        e = np.exp(-np.abs(self.data))
        s = np.where(self.data >= 0, 1 / (1 + e), e / (1 + e))
        return self._unary(s, s * (1 - s), "sigmoid")

    # --- reductions and shape ops ---------------------------------------------------------

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward() -> None:
            g = out.grad
            if axis is not None and not keepdims:  # put the summed axes back as size 1
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.shape))

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        n = (
            self.data.size
            if axis is None
            else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        )
        return self.sum(axis, keepdims) / float(n)

    def max(self, axis: int, keepdims: bool = False) -> Tensor:
        """Gradient flows only to the maximum (split evenly between ties)."""
        m = self.data.max(axis=axis, keepdims=True)
        out = self._make(m if keepdims else np.squeeze(m, axis), (self,), "max")

        def _backward() -> None:
            mask = (self.data == m).astype(float)
            mask /= mask.sum(axis=axis, keepdims=True)
            g = out.grad if keepdims else np.expand_dims(out.grad, axis)
            self._accumulate(mask * g)

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        out = self._make(self.data.reshape(*shape), (self,), "reshape")

        def _backward() -> None:
            self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    @property
    def T(self) -> Tensor:
        out = self._make(self.data.T, (self,), "T")

        def _backward() -> None:
            self._accumulate(out.grad.T)

        out._backward = _backward
        return out

    def __getitem__(self, index) -> Tensor:
        out = self._make(self.data[index], (self,), "getitem")

        def _backward() -> None:
            g = np.zeros_like(self.data)
            np.add.at(g, index, out.grad)  # add.at handles repeated indices correctly
            self._accumulate(g)

        out._backward = _backward
        return out

    # --- composite ops used by networks ---------------------------------------------------------

    def log_softmax(self, axis: int = -1) -> Tensor:
        """x - logsumexp(x), with the max subtracted for stability (max is a constant shift)."""
        shifted = self - Tensor(self.data.max(axis=axis, keepdims=True))
        return shifted - shifted.exp().sum(axis=axis, keepdims=True).log()

    def softmax(self, axis: int = -1) -> Tensor:
        return self.log_softmax(axis).exp()

    # --- backward pass ---------------------------------------------------------------------------

    def backward(self, grad: np.ndarray | None = None) -> None:
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("backward() without a gradient needs a scalar output")
            grad = np.ones_like(self.data)
        order: list[Tensor] = []
        seen: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend(
                    (c, False) for c in node._prev if c.requires_grad and id(c) not in seen
                )
        self.grad = np.asarray(grad, dtype=float)
        for node in reversed(order):
            node._backward()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean negative log-likelihood of integer targets from raw logits."""
    logp = logits.log_softmax(axis=-1)
    return -logp[np.arange(len(targets)), targets].mean()


def mse(pred: Tensor, target: np.ndarray) -> Tensor:
    return ((pred - Tensor(target)) ** 2).mean()


class SGD:
    """Minimal optimizer: p <- p - lr * (grad + weight_decay * p), with optional momentum."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        self.params = list(params)
        self.lr, self.momentum, self.weight_decay = lr, momentum, weight_decay
        self._velocity = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        for p, v in zip(self.params, self._velocity):
            if p.grad is None:
                continue
            g = p.grad + self.weight_decay * p.data
            v *= self.momentum
            v += g
            p.data -= self.lr * v


class Adam:
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        """weight_decay here is DECOUPLED (AdamW): applied to the weights directly, not via the gradient."""
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.weight_decay = lr, betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            m[:] = self.b1 * m + (1 - self.b1) * p.grad
            v[:] = self.b2 * v + (1 - self.b2) * p.grad**2
            m_hat = m / (1 - self.b1**self.t)
            v_hat = v / (1 - self.b2**self.t)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)
