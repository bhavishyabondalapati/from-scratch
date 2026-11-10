from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits, make_moons

from s2_math.gradient_check.impl import check_gradient, numerical_gradient, relative_error

from .impl import (
    ACTIVATIONS,
    Linear,
    SoftmaxCrossEntropy,
    activation_stats,
    he_init,
    make_mlp,
    train_classifier,
    xavier_init,
)


@pytest.mark.parametrize("name", ACTIVATIONS)
def test_activation_backward_matches_finite_differences(name, rng):
    layer = ACTIVATIONS[name]()
    x = rng.normal(size=(4, 5))
    x[np.abs(x) < 1e-3] = 0.5  # stay away from the ReLU kink
    upstream = rng.normal(size=(4, 5))
    layer.forward(x)
    analytic = layer.backward(upstream)
    numeric = numerical_gradient(lambda z: float(np.sum(layer.forward(z) * upstream)), x)
    assert relative_error(analytic, numeric) < 1e-7


def test_gelu_matches_pytorch(rng):
    x = rng.normal(size=50) * 3
    ours = ACTIVATIONS["gelu"]().forward(x)
    ref = torch.nn.functional.gelu(torch.tensor(x), approximate="tanh").numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-12)


@pytest.mark.parametrize("activation", ["relu", "tanh", "gelu"])
def test_full_network_gradients_pass_gradient_check(activation, rng):
    model = make_mlp([5, 7, 6, 3], activation, seed=1)
    X, y = rng.normal(size=(8, 5)), rng.integers(0, 3, size=8)
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(model(X), y)
    grad_input = model.backward(loss_fn.backward())
    for layer, name in model.parameters():
        param = layer.params[name]
        analytic = layer.grads[name].copy()
        ok, err = check_gradient(lambda _: loss_fn.forward(model(X), y), analytic, param)
        assert ok, (name, err)
    ok, err = check_gradient(lambda z: loss_fn.forward(model(z), y), grad_input, X)
    assert ok, err


def test_matches_pytorch_forward_and_backward(rng):
    model = make_mlp([4, 6, 3], "tanh", init="xavier", seed=2)
    X, y = rng.normal(size=(5, 4)), rng.integers(0, 3, size=5)
    loss_fn = SoftmaxCrossEntropy()
    loss = loss_fn.forward(model(X), y)
    model.backward(loss_fn.backward())
    lin1, lin2 = model.layers[0], model.layers[2]
    W1 = torch.tensor(lin1.params["W"], requires_grad=True)
    b1 = torch.tensor(lin1.params["b"], requires_grad=True)
    W2 = torch.tensor(lin2.params["W"], requires_grad=True)
    b2 = torch.tensor(lin2.params["b"], requires_grad=True)
    logits = torch.tanh(torch.tensor(X) @ W1 + b1) @ W2 + b2
    ref = torch.nn.functional.cross_entropy(logits, torch.tensor(y))
    ref.backward()
    assert loss == pytest.approx(ref.item())
    for ours, theirs in [
        (lin1.grads["W"], W1),
        (lin1.grads["b"], b1),
        (lin2.grads["W"], W2),
        (lin2.grads["b"], b2),
    ]:
        np.testing.assert_allclose(ours, theirs.grad.numpy(), atol=1e-12)


def test_initialisation_variances():
    rng = np.random.default_rng(0)
    W = xavier_init(400, 600, rng)
    assert W.var() == pytest.approx(2 / (400 + 600), rel=0.05)
    W = he_init(400, 600, rng)
    assert W.var() == pytest.approx(2 / 400, rel=0.05)


def test_he_init_keeps_relu_signal_alive_in_deep_nets():
    sizes = [100] + [100] * 20 + [10]
    he = activation_stats(sizes, "relu", "he")
    xavier = activation_stats(sizes, "relu", "xavier")
    assert 0.3 < he[-1] < 3.0  # still O(1) after 20 ReLU layers
    assert xavier[-1] < 0.01 * xavier[0]  # shrinks by ~sqrt(2) per layer and vanishes
    tanh = activation_stats(sizes, "tanh", "xavier")
    assert tanh[-1] > 0.05  # xavier is the right match for tanh


def test_learns_xor_shaped_moons_better_than_linear():
    X, y = make_moons(400, noise=0.1, random_state=0)
    linear = make_mlp([2, 2], seed=0)
    train_classifier(linear, X, y, epochs=60, lr=0.1)
    deep = make_mlp([2, 32, 32, 2], "relu", seed=0)
    train_classifier(deep, X, y, epochs=60, lr=0.1)
    acc = lambda m: np.mean(m(X).argmax(1) == y)  # noqa: E731
    assert acc(deep) > 0.98 > acc(linear) + 0.05


def test_digits_classifier_and_training_curve():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    idx = np.random.default_rng(0).permutation(len(y))
    tr, te = idx[:1400], idx[1400:]
    model = make_mlp([64, 128, 64, 10], "relu", seed=0)
    history = train_classifier(model, X[tr], y[tr], epochs=40, lr=0.05)
    test_acc = np.mean(model(X[te]).argmax(1) == y[te])
    assert history[-1] < 0.1 * history[0]
    assert test_acc > 0.96

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history)
    ax.set(
        xlabel="epoch",
        ylabel="train cross-entropy",
        yscale="log",
        title=f"MLP 64-128-64-10 on digits (test acc {test_acc:.3f})",
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)


def test_linear_layer_shapes(rng):
    layer = Linear(3, 2, rng=rng)
    out = layer.forward(rng.normal(size=(7, 3)))
    assert out.shape == (7, 2)
    assert layer.backward(np.ones((7, 2))).shape == (7, 3)
    np.testing.assert_allclose(layer.grads["b"], [7, 7])
