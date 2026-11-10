from pathlib import Path

import numpy as np
import pytest
import torch

from s2_math.gradient_check.impl import check_gradient
from s4_deep_learning.mlp.impl import Linear, ReLU, Sequential, train_classifier

from .impl import BatchNorm1d, LayerNorm


def with_random_affine(layer, rng):
    layer.params["gamma"] = rng.normal(size=layer.params["gamma"].shape)
    layer.params["beta"] = rng.normal(size=layer.params["beta"].shape)
    return layer


@pytest.mark.parametrize("cls", [BatchNorm1d, LayerNorm])
def test_backward_passes_gradient_check(cls, rng):
    layer = with_random_affine(cls(5), rng)
    x = rng.normal(size=(8, 5)) * 3 + 1
    upstream = rng.normal(size=(8, 5))

    def loss(z):
        return float(np.sum(layer.forward(z) * upstream))

    layer.forward(x)
    dx = layer.backward(upstream)
    analytic = {k: v.copy() for k, v in layer.grads.items()}
    ok, err = check_gradient(loss, dx, x)
    assert ok, err
    for name in ("gamma", "beta"):
        ok, err = check_gradient(lambda _: loss(x), analytic[name], layer.params[name])
        assert ok, (name, err)


def test_batchnorm_matches_pytorch_train_and_eval(rng):
    ours = BatchNorm1d(4, momentum=0.1)
    ref = torch.nn.BatchNorm1d(4, momentum=0.1).double()
    for _ in range(5):
        x = rng.normal(size=(16, 4)) * 2 + 3
        g = rng.normal(size=(16, 4))
        out = ours.forward(x)
        dx = ours.backward(g)
        xt = torch.tensor(x, requires_grad=True)
        ref.zero_grad()
        rout = ref(xt)
        rout.backward(torch.tensor(g))
        np.testing.assert_allclose(out, rout.detach().numpy(), atol=1e-10)
        np.testing.assert_allclose(dx, xt.grad.numpy(), atol=1e-10)
        np.testing.assert_allclose(ours.grads["gamma"], ref.weight.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.running_mean, ref.running_mean.numpy())
    np.testing.assert_allclose(ours.running_var, ref.running_var.numpy())
    ours.training = False
    ref.eval()
    x = rng.normal(size=(3, 4))
    np.testing.assert_allclose(ours.forward(x), ref(torch.tensor(x)).detach().numpy(), atol=1e-10)


def test_layernorm_matches_pytorch_including_3d_input(rng):
    ours = with_random_affine(LayerNorm(6), rng)
    ref = torch.nn.LayerNorm(6).double()
    with torch.no_grad():
        ref.weight.copy_(torch.tensor(ours.params["gamma"]))
        ref.bias.copy_(torch.tensor(ours.params["beta"]))
    x = rng.normal(size=(2, 5, 6))  # (batch, sequence, features) as in a transformer
    g = rng.normal(size=(2, 5, 6))
    out = ours.forward(x)
    dx = ours.backward(g)
    xt = torch.tensor(x, requires_grad=True)
    rout = ref(xt)
    rout.backward(torch.tensor(g))
    np.testing.assert_allclose(out, rout.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(dx, xt.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.grads["gamma"], ref.weight.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.grads["beta"], ref.bias.grad.numpy(), atol=1e-10)


def test_normalised_outputs_have_zero_mean_unit_variance(rng):
    x = rng.normal(size=(64, 10)) * 7 - 4
    bn, ln = BatchNorm1d(10), LayerNorm(10)
    np.testing.assert_allclose(bn.forward(x).mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(bn.forward(x).std(axis=0), 1, atol=1e-3)
    np.testing.assert_allclose(ln.forward(x).mean(axis=1), 0, atol=1e-10)
    np.testing.assert_allclose(ln.forward(x).std(axis=1), 1, atol=1e-3)
    # LayerNorm treats each sample independently: the batch does not matter.
    np.testing.assert_allclose(ln.forward(x[:1]), ln.forward(x)[:1])


def deep_net(norm, seed=0, depth=8, width=64):
    rng = np.random.default_rng(seed)
    layers = [Linear(2, width, init="xavier", rng=rng)]
    for _ in range(depth):
        if norm:
            layers.append(norm(width))
        layers += [ReLU(), Linear(width, width, init="xavier", rng=rng)]
    layers.append(Linear(width, 2, init="xavier", rng=rng))
    return Sequential(*layers)


@pytest.mark.slow
def test_normalisation_speeds_up_training_of_a_deep_net():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 2))
    y = (np.sin(2 * X[:, 0]) + X[:, 1] ** 2 > 0.8).astype(int)
    curves = {}
    for name, norm in [("no norm", None), ("batch norm", BatchNorm1d), ("layer norm", LayerNorm)]:
        model = deep_net(norm)  # 8 hidden layers with (deliberately poor) Xavier init for ReLU
        curves[name] = train_classifier(model, X, y, epochs=25, lr=0.02)
    assert curves["batch norm"][-1] < 0.5 * curves["no norm"][-1]
    assert curves["layer norm"][-1] < 0.5 * curves["no norm"][-1]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    for name, c in curves.items():
        ax.plot(c, label=f"{name} (final {c[-1]:.3f})")
    ax.set(xlabel="epoch", ylabel="train loss", yscale="log", title="9-layer ReLU MLP, same lr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
