from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits

from s2_math.gradient_check.impl import relative_error

from .impl import SGD, Adam, Tensor, cross_entropy, mse


def run_both(fn, *arrays):
    """Evaluate fn with our Tensors and with torch tensors; return outputs and input grads."""
    ours = [Tensor(a, requires_grad=True) for a in arrays]
    out = fn(*ours)
    out.backward()
    theirs = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrays]
    ref = fn(*theirs)
    ref.backward()
    return out, ref, ours, theirs


CASES = {
    "broadcast add/mul": (lambda a, b: ((a + b) * a).sum(), [(3, 4), (4,)]),
    "keepdims broadcast": (lambda a, b: (a * b).mean(), [(3, 4), (3, 1)]),
    "division": (lambda a, b: (a / (b * b + 1.0)).sum(), [(2, 5), (2, 5)]),
    "matmul 2d": (lambda a, b: (a @ b).tanh().sum(), [(3, 4), (4, 5)]),
    "matmul batched": (lambda a, b: (a @ b).sum(), [(2, 3, 4), (2, 4, 5)]),
    "matrix @ vector": (lambda a, b: ((a @ b) ** 2).sum(), [(3, 4), (4,)]),
    "vector @ matrix": (lambda a, b: ((a @ b) ** 2).sum(), [(4,), (4, 3)]),
    "sum over axis": (lambda a, b: (a.sum(axis=0) * b).sum(), [(3, 4), (4,)]),
    "exp/log/sigmoid": (
        lambda a, b: ((a * 0.3).exp() + (b * b + 1).log() + a.sigmoid()).sum(),
        [(3, 3), (3, 3)],
    ),
    "relu": (lambda a, b: (a.relu() * b).sum(), [(5, 4), (5, 4)]),
    "transpose/reshape": (lambda a, b: (a.T.reshape(2, 6) * b).sum(), [(3, 4), (2, 6)]),
}


@pytest.mark.parametrize("name", CASES)
def test_gradients_match_pytorch(name, rng):
    fn, shapes = CASES[name]
    arrays = [rng.normal(size=s) for s in shapes]
    out, ref, ours, theirs = run_both(fn, *arrays)
    assert out.data == pytest.approx(ref.item())
    for o, t in zip(ours, theirs):
        assert o.grad.shape == o.shape
        assert relative_error(o.grad, t.grad.numpy()) < 1e-10


def test_softmax_max_indexing_against_pytorch(rng):
    x = rng.normal(size=(4, 6)) * 3
    targets = np.array([0, 5, 2, 2])
    t = Tensor(x, requires_grad=True)
    loss = cross_entropy(t, targets) + t.max(axis=1).sum() * 0.1 + t.softmax()[:, 1].sum()
    loss.backward()
    tt = torch.tensor(x, requires_grad=True)
    ref = (
        torch.nn.functional.cross_entropy(tt, torch.tensor(targets))
        + tt.max(dim=1).values.sum() * 0.1
        + tt.softmax(dim=-1)[:, 1].sum()
    )
    ref.backward()
    assert loss.data == pytest.approx(ref.item())
    np.testing.assert_allclose(t.grad, tt.grad.numpy(), atol=1e-12)


def test_repeated_index_gradients_accumulate():
    x = Tensor(np.arange(4.0), requires_grad=True)
    x[np.array([1, 1, 3])].sum().backward()
    np.testing.assert_array_equal(x.grad, [0, 2, 0, 1])


def test_backward_requires_scalar_or_explicit_grad(rng):
    x = Tensor(rng.normal(size=3), requires_grad=True)
    with pytest.raises(RuntimeError):
        (x * 2).backward()
    (x * 2).backward(np.ones(3))
    np.testing.assert_array_equal(x.grad, [2, 2, 2])
    constant = Tensor(np.ones(3))
    (constant * x).sum().backward()
    assert constant.grad is None  # no gradient tracked for constants


def test_optimizers_match_pytorch(rng):
    w0 = rng.normal(size=(3, 2))
    X = rng.normal(size=(10, 3))
    y = rng.normal(size=(10, 2))
    for ours_cls, torch_cls, kw, tkw in [
        (
            SGD,
            torch.optim.SGD,
            dict(lr=0.1, momentum=0.9, weight_decay=0.01),
            dict(lr=0.1, momentum=0.9, weight_decay=0.01),
        ),
        (Adam, torch.optim.AdamW, dict(lr=0.05, weight_decay=0.1), dict(lr=0.05, weight_decay=0.1)),
    ]:
        w = Tensor(w0.copy(), requires_grad=True)
        wt = torch.tensor(w0.copy(), requires_grad=True)
        opt, topt = ours_cls([w], **kw), torch_cls([wt], **tkw)
        for _ in range(20):
            opt.zero_grad()
            mse(Tensor(X) @ w, y).backward()
            opt.step()
            topt.zero_grad()
            ((torch.tensor(X) @ wt - torch.tensor(y)) ** 2).mean().backward()
            topt.step()
        np.testing.assert_allclose(w.data, wt.detach().numpy(), rtol=1e-8, atol=1e-10)


@pytest.mark.slow
def test_softmax_regression_on_digits_beats_baseline():
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[:1400], idx[1400:]
    W = Tensor(np.zeros((64, 10)), requires_grad=True)
    b = Tensor(np.zeros(10), requires_grad=True)
    opt = Adam([W, b], lr=0.05)
    losses = []
    for _ in range(150):
        opt.zero_grad()
        loss = cross_entropy(Tensor(X[tr]) @ W + b, y[tr])
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    acc = np.mean(((X[te] @ W.data + b.data).argmax(1)) == y[te])
    assert acc > 0.93 > np.bincount(y[te]).max() / len(te)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(losses)
    ax.set(
        xlabel="step",
        ylabel="cross-entropy",
        title=f"tensor autograd softmax reg. (test acc {acc:.3f})",
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
