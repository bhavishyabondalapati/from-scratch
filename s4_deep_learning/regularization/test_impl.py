from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import load_digits

from s4_deep_learning.mlp.impl import Linear, ReLU, Sequential, make_mlp

from .impl import (
    Dropout,
    constant_lr,
    cosine_with_warmup,
    exponential_decay,
    step_decay,
    train,
    weight_norm,
)


def test_dropout_preserves_expectation_and_is_off_at_eval():
    x = np.ones((2000, 50))
    d = Dropout(p=0.3, seed=0)
    out = d.forward(x)
    assert np.mean(out == 0) == pytest.approx(0.3, abs=0.01)
    assert out.mean() == pytest.approx(1.0, abs=0.01)  # inverted scaling keeps the mean
    assert set(np.unique(out)) == {0.0, 1 / 0.7}
    d.training = False
    np.testing.assert_array_equal(d.forward(x), x)
    with pytest.raises(ValueError):
        Dropout(1.0)


def test_dropout_backward_uses_the_same_mask(rng):
    d = Dropout(0.5, seed=1)
    x = rng.normal(size=(6, 4))
    out = d.forward(x)
    g = rng.normal(size=(6, 4))
    grad = d.backward(g)
    np.testing.assert_allclose(
        grad, g * out / np.where(x == 0, 1, x)
    )  # same units zeroed and scaled
    # Gradient check with the mask frozen: f(x) = sum(dropout(x) * g) is linear in x.
    np.testing.assert_allclose(grad, d.mask * g)


def test_schedules_match_pytorch():
    total, warmup, base = 100, 10, 0.5
    ours = cosine_with_warmup(base, total, warmup)
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [
            torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - warmup),
        ],
        milestones=[warmup],
    )
    for step in range(total):
        assert ours(step) == pytest.approx(opt.param_groups[0]["lr"], abs=1e-9)
        opt.step()
        sched.step()
    s = step_decay(1.0, 0.1, 30)
    assert (s(0), s(29), s(30), s(65)) == (1.0, 1.0, pytest.approx(0.1), pytest.approx(0.01))
    assert exponential_decay(2.0, 0.5)(3) == 0.25
    assert constant_lr(0.3)(1000) == 0.3


@pytest.fixture(scope="module")
def small_digits():
    """Few training samples + a big net = an easy model to overfit."""
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    idx = np.random.default_rng(0).permutation(len(y))
    return X[idx[:150]], y[idx[:150]], X[idx[600:]], y[idx[600:]]


def net(dropout: float = 0.0, seed: int = 0) -> Sequential:
    rng = np.random.default_rng(seed)
    layers = [Linear(64, 256, rng=rng), ReLU()]
    if dropout:
        layers.append(Dropout(dropout, seed=seed))
    layers += [Linear(256, 256, rng=rng), ReLU()]
    if dropout:
        layers.append(Dropout(dropout, seed=seed + 1))
    layers.append(Linear(256, 10, rng=rng))
    return Sequential(*layers)


def test_weight_decay_shrinks_weights(small_digits):
    Xtr, ytr, _, _ = small_digits
    plain, decayed = make_mlp([64, 64, 10], seed=0), make_mlp([64, 64, 10], seed=0)
    train(plain, Xtr, ytr, 30, constant_lr(0.05))
    train(decayed, Xtr, ytr, 30, constant_lr(0.05), weight_decay=0.05)
    assert weight_norm(decayed) < 0.7 * weight_norm(plain)


@pytest.mark.slow
def test_regularisation_reduces_overfitting_and_saves_curves(small_digits):
    Xtr, ytr, Xva, yva = small_digits
    runs = {
        "none": (net(), {}),
        "dropout 0.5": (net(0.5), {}),
        "weight decay 1e-2": (net(), {"weight_decay": 1e-2}),
    }
    results = {}
    for name, (model, kw) in runs.items():
        sched = cosine_with_warmup(0.05, total=150 * 3, warmup=20)
        results[name] = train(
            model, Xtr, ytr, 150, sched, X_val=Xva, y_val=yva, batch_size=50, **kw
        )
    final = {k: np.mean(v["val_acc"][-10:]) for k, v in results.items()}
    assert results["none"]["train_loss"][-1] < 1e-2  # memorised the 150 training images
    assert final["dropout 0.5"] > final["none"]
    assert final["weight decay 1e-2"] > final["none"]
    assert min(final.values()) > 0.8  # all far above the 10% baseline

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, h in results.items():
        axes[0].plot(h["train_loss"], label=name)
        axes[1].plot(h["val_acc"], label=f"{name} ({final[name]:.3f})")
    axes[0].set(yscale="log", xlabel="epoch", ylabel="train loss")
    axes[1].set(xlabel="epoch", ylabel="val accuracy", ylim=(0.6, 1.0))
    axes[1].legend(fontsize=8)
    fig.suptitle("150 training digits, 64-256-256-10 MLP")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
