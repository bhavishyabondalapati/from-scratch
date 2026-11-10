"""A scalar automatic-differentiation engine (micrograd-style) and a tiny MLP on top of it.

Every Value remembers which Values produced it and how. Calling .backward() on
the final result walks that graph in reverse topological order and applies the
chain rule at every node.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable


class Value:
    """A single number that tracks its gradient."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0  # d(final output) / d(this value), filled in by backward()
        self._backward: Callable[[], None] = lambda: None  # pushes self.grad to the children
        self._prev = _children
        self._op = _op  # for debugging / drawing the graph

    # --- operations -----------------------------------------------------------
    # Each one computes the forward value and defines a closure that applies the
    # LOCAL derivative. Gradients are ACCUMULATED with += because a value may be
    # used in several places (the multivariable chain rule sums over all uses).

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad  # d(a+b)/da = 1
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad  # d(a*b)/da = b
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        if not isinstance(exponent, int | float):
            raise TypeError("only constant exponents are supported")
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad  # d e^x / dx = e^x

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        s = (
            1 / (1 + math.exp(-self.data))
            if self.data >= 0
            else math.exp(self.data) / (1 + math.exp(self.data))
        )
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    # Everything else is built from the primitives above.
    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other if isinstance(other, Value) else Value(other)) ** -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __rsub__(self, other: float) -> Value:
        return Value(other) - self

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __rtruediv__(self, other: float) -> Value:
        return Value(other) / self

    # --- reverse-mode differentiation ------------------------------------------

    def backward(self) -> None:
        """Fill in .grad for every Value this one depends on.

        Topological order guarantees that when we process a node, every node
        that USES it has already pushed its gradient contribution.
        """
        order: list[Value] = []
        seen: set[int] = set()

        def build(v: Value) -> None:  # iterative DFS (deep graphs would hit the recursion limit)
            stack = [(v, False)]
            while stack:
                node, expanded = stack.pop()
                if expanded:
                    order.append(node)
                    continue
                if id(node) in seen:
                    continue
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((child, False) for child in node._prev if id(child) not in seen)

        build(self)
        self.grad = 1.0  # d(self)/d(self)
        for node in reversed(order):
            node._backward()

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"


# --- A tiny neural-network library on top of Value --------------------------------


class Neuron:
    def __init__(
        self, n_inputs: int, nonlinear: bool = True, rng: random.Random | None = None
    ) -> None:
        rng = rng or random.Random(0)
        scale = 1 / math.sqrt(n_inputs)
        self.w = [Value(rng.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.nonlinear = nonlinear

    def __call__(self, x: Iterable[Value | float]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlinear else act

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class Layer:
    def __init__(self, n_in: int, n_out: int, nonlinear: bool, rng: random.Random) -> None:
        self.neurons = [Neuron(n_in, nonlinear, rng) for _ in range(n_out)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Fully connected net: tanh hidden layers, linear output layer."""

    def __init__(self, n_in: int, layer_sizes: list[int], seed: int = 0) -> None:
        rng = random.Random(seed)
        sizes = [n_in, *layer_sizes]
        self.layers = [
            Layer(sizes[i], sizes[i + 1], nonlinear=i < len(layer_sizes) - 1, rng=rng)
            for i in range(len(layer_sizes))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0
