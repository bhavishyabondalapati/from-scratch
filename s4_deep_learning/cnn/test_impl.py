from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from sklearn.datasets import load_digits

from s2_math.gradient_check.impl import check_gradient
from s4_deep_learning.mlp.impl import SoftmaxCrossEntropy, make_mlp, train_classifier

from .impl import AvgPool2d, Conv2d, MaxPool2d, col2im, conv2d_naive, im2col, small_cnn


@pytest.mark.parametrize("stride, pad, k", [(1, 0, 3), (1, 1, 3), (2, 1, 3), (2, 0, 2), (1, 2, 5)])
def test_conv_forward_matches_naive_and_pytorch(rng, stride, pad, k):
    x = rng.normal(size=(2, 3, 9, 8))
    layer = Conv2d(3, 4, k, stride, pad, rng=rng)
    layer.params["b"] = rng.normal(size=4)
    out = layer.forward(x)
    np.testing.assert_allclose(
        out, conv2d_naive(x, layer.params["W"], layer.params["b"], stride, pad), atol=1e-10
    )
    ref = F.conv2d(
        torch.tensor(x),
        torch.tensor(layer.params["W"]),
        torch.tensor(layer.params["b"]),
        stride,
        pad,
    )
    np.testing.assert_allclose(out, ref.numpy(), atol=1e-10)


@pytest.mark.parametrize("stride, pad", [(1, 1), (2, 0), (2, 1)])
def test_conv_backward_matches_pytorch(rng, stride, pad):
    x = rng.normal(size=(2, 3, 7, 7))
    layer = Conv2d(3, 5, 3, stride, pad, rng=rng)
    out = layer.forward(x)
    g = rng.normal(size=out.shape)
    dx = layer.backward(g)
    xt = torch.tensor(x, requires_grad=True)
    wt = torch.tensor(layer.params["W"], requires_grad=True)
    bt = torch.tensor(layer.params["b"], requires_grad=True)
    F.conv2d(xt, wt, bt, stride, pad).backward(torch.tensor(g))
    np.testing.assert_allclose(dx, xt.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(layer.grads["W"], wt.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(layer.grads["b"], bt.grad.numpy(), atol=1e-10)


def test_conv_backward_gradient_check(rng):
    x = rng.normal(size=(1, 2, 5, 5))
    layer = Conv2d(2, 3, 3, stride=2, pad=1, rng=rng)
    g = rng.normal(size=layer.forward(x).shape)
    dx = layer.backward(g)
    ok, err = check_gradient(lambda z: float(np.sum(layer.forward(z) * g)), dx, x)
    assert ok, err


def test_col2im_is_the_adjoint_of_im2col(rng):
    # <im2col(x), c> == <x, col2im(c)> for all x, c: the defining property of a correct backward.
    x = rng.normal(size=(2, 3, 6, 5))
    cols = im2col(x, 3, 3, stride=2, pad=1)
    c = rng.normal(size=cols.shape)
    assert np.sum(cols * c) == pytest.approx(np.sum(x * col2im(c, x.shape, 3, 3, 2, 1)))


@pytest.mark.parametrize("cls, ref_fn", [(MaxPool2d, F.max_pool2d), (AvgPool2d, F.avg_pool2d)])
def test_pooling_matches_pytorch(rng, cls, ref_fn):
    x = rng.normal(size=(2, 3, 8, 6))
    layer = cls(2)
    out = layer.forward(x)
    g = rng.normal(size=out.shape)
    dx = layer.backward(g)
    xt = torch.tensor(x, requires_grad=True)
    ref = ref_fn(xt, 2)
    ref.backward(torch.tensor(g))
    np.testing.assert_allclose(out, ref.detach().numpy())
    np.testing.assert_allclose(dx, xt.grad.numpy())


def test_whole_cnn_gradient_check(rng):
    model = small_cnn(seed=1)
    X = rng.normal(size=(2, 1, 8, 8))
    y = np.array([3, 7])
    loss_fn = SoftmaxCrossEntropy()
    loss_fn.forward(model(X), y)
    model.backward(loss_fn.backward())
    conv1 = model.layers[0]
    analytic = conv1.grads["W"].copy()
    ok, err = check_gradient(
        lambda _: loss_fn.forward(model(X), y), analytic, conv1.params["W"], tol=1e-5
    )
    assert ok, err


def shift(images, dy, dx):
    """Move every image by (dy, dx) pixels, filling the gap with zeros."""
    out = np.zeros_like(images)
    H, W = images.shape[-2:]
    src_y, dst_y = slice(max(-dy, 0), H - max(dy, 0)), slice(max(dy, 0), H - max(-dy, 0))
    src_x, dst_x = slice(max(-dx, 0), W - max(dx, 0)), slice(max(dx, 0), W - max(-dx, 0))
    out[..., dst_y, dst_x] = images[..., src_y, src_x]
    return out


@pytest.mark.slow
def test_cnn_on_digits_and_robustness_to_shifts():
    X, y = load_digits(return_X_y=True)
    X = (X / 16.0).reshape(-1, 1, 8, 8)
    idx = np.random.default_rng(0).permutation(len(y))
    tr, te = idx[:1400], idx[1400:]
    cnn = small_cnn(seed=0)
    history = train_classifier(cnn, X[tr], y[tr], epochs=15, lr=0.05, batch_size=32)
    mlp = make_mlp([64, 64, 10], seed=0)
    train_classifier(mlp, X[tr].reshape(len(tr), -1), y[tr], epochs=15, lr=0.05, batch_size=32)

    def acc(images):
        return (
            float(np.mean(cnn(images).argmax(1) == y[te])),
            float(np.mean(mlp(images.reshape(len(te), -1)).argmax(1) == y[te])),
        )

    cnn_acc, mlp_acc = acc(X[te])
    assert cnn_acc > 0.97 and mlp_acc > 0.97  # both learn the centred digits well
    shifted = [acc(shift(X[te], dy, dx)) for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]]
    cnn_shift = np.mean([c for c, _ in shifted])
    mlp_shift = np.mean([m for _, m in shifted])
    # Weight sharing + pooling make the CNN far more tolerant of a 1-pixel shift.
    assert cnn_shift > mlp_shift + 0.15

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history)
    ax.set(
        xlabel="epoch",
        ylabel="train loss",
        yscale="log",
        title=f"NumPy CNN on digits: test {cnn_acc:.3f}, shifted {cnn_shift:.2f}",
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print(f"cnn {cnn_acc:.3f}/{cnn_shift:.3f}  mlp {mlp_acc:.3f}/{mlp_shift:.3f}")
