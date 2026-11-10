import math
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_moons

from .impl import MLP, Value


def test_simple_gradients_by_hand():
    a, b = Value(2.0), Value(-3.0)
    c = a * b + a  # dc/da = b + 1 = -2, dc/db = a = 2
    c.backward()
    assert (a.grad, b.grad) == (-2.0, 2.0)
    x = Value(3.0)
    y = x * x * x  # a value used several times: gradients must accumulate
    y.backward()
    assert x.grad == pytest.approx(27.0)


def test_matches_pytorch_on_a_complicated_expression():
    def build(a, b, lib):
        relu = (lambda v: v.relu()) if lib == "ours" else torch.relu
        tanh = (lambda v: v.tanh()) if lib == "ours" else torch.tanh
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + relu(b + a)
        d = d + 3 * d + relu(b - a)
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        h = (
            tanh(g / 50) + (a * 0.1).exp() + (b * b + 1).log() + (a / 3).sigmoid()
            if lib == "ours"
            else (tanh(g / 50) + torch.exp(a * 0.1) + torch.log(b * b + 1) + torch.sigmoid(a / 3))
        )
        return h

    a, b = Value(-4.0), Value(2.0)
    out = build(a, b, "ours")
    out.backward()
    at = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    ref = build(at, bt, "torch")
    ref.backward()
    assert out.data == pytest.approx(ref.item(), rel=1e-12)
    assert a.grad == pytest.approx(at.grad.item(), rel=1e-9)
    assert b.grad == pytest.approx(bt.grad.item(), rel=1e-9)


@pytest.mark.parametrize("op", ["exp", "log", "tanh", "relu", "sigmoid", "pow", "div"])
def test_every_primitive_against_finite_differences(op):
    def f(x: float):
        v = Value(x)
        out = {
            "exp": lambda: v.exp(),
            "log": lambda: v.log(),
            "tanh": lambda: v.tanh(),
            "relu": lambda: v.relu(),
            "sigmoid": lambda: v.sigmoid(),
            "pow": lambda: v**2.5,
            "div": lambda: 3.0 / v,
        }[op]()
        return v, out

    for x in (0.3, 1.7, 2.9):
        v, out = f(x)
        out.backward()
        eps = 1e-6
        numeric = (f(x + eps)[1].data - f(x - eps)[1].data) / (2 * eps)
        assert v.grad == pytest.approx(numeric, rel=1e-5, abs=1e-8)


def test_deep_chain_does_not_hit_recursion_limit():
    x = Value(1.0)
    y = x
    for _ in range(5000):
        y = y * 1.0 + 0.0
    y.backward()
    assert x.grad == pytest.approx(1.0)


def test_mlp_gradients_match_pytorch():
    model = MLP(3, [4, 4, 1], seed=0)
    x = [0.5, -1.2, 2.0]
    out = model(x)
    out.backward()
    # Rebuild the same network in PyTorch with identical weights.
    xt = torch.tensor(x, dtype=torch.float64)
    h = xt
    params = []
    for i, layer in enumerate(model.layers):
        W = torch.tensor(
            [[w.data for w in n.w] for n in layer.neurons], dtype=torch.float64, requires_grad=True
        )
        b = torch.tensor([n.b.data for n in layer.neurons], dtype=torch.float64, requires_grad=True)
        params.append((W, b))
        h = W @ h + b
        if i < len(model.layers) - 1:
            h = torch.tanh(h)
    h.sum().backward()
    assert out.data == pytest.approx(h.item())
    for layer, (W, b) in zip(model.layers, params):
        ours = np.array([[w.grad for w in n.w] for n in layer.neurons])
        np.testing.assert_allclose(ours, W.grad.numpy(), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose([n.b.grad for n in layer.neurons], b.grad.numpy(), atol=1e-12)


@pytest.mark.slow
def test_trains_a_classifier_on_moons():
    X, y = make_moons(100, noise=0.1, random_state=0)
    y = 2 * y - 1  # labels in {-1, +1}
    model = MLP(2, [16, 16, 1], seed=1)
    losses = []
    for step in range(60):
        scores = [model(list(map(float, xi))) for xi in X]
        # SVM-style hinge loss + L2, as in the original micrograd demo.
        data_loss = sum(((1 + -yi * s).relu() for yi, s in zip(y, scores)), Value(0.0)) / len(X)
        reg = 1e-4 * sum((p * p for p in model.parameters()), Value(0.0))
        loss = data_loss + reg
        model.zero_grad()
        loss.backward()
        lr = 1.0 - 0.9 * step / 60
        for p in model.parameters():
            p.data -= lr * p.grad
        losses.append(loss.data)
    scores = [model(list(map(float, xi))) for xi in X]  # evaluate AFTER the last update
    accuracy = np.mean([(s.data > 0) == (yi > 0) for s, yi in zip(scores, y)])
    # Majority class is 50%; a linear classifier gets about 85% on this data.
    assert accuracy > 0.9 and losses[-1] < 0.5 * losses[0]
    _save_curve(losses, accuracy)


def _save_curve(losses, accuracy):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses)
    ax.set(xlabel="step", ylabel="loss", title=f"scalar autograd MLP on moons (acc {accuracy:.2f})")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    assert math.isfinite(losses[-1])
