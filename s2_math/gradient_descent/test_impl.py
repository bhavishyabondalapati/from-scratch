import numpy as np
import pytest
import torch

from .impl import SGD, Adam, Momentum, RMSProp, minibatch_sgd, minimize

# An ill-conditioned bowl: steep in one direction, shallow in the other.
A = np.array([[10.0, 0.0], [0.0, 1.0]])
B = np.array([1.0, -2.0])
X_STAR = np.linalg.solve(A, B)  # the minimiser of 0.5 x^T A x - b^T x


def quad_loss(x):
    return 0.5 * x @ A @ x - B @ x


def quad_grad(x):
    return A @ x - B


PAIRS = [
    (lambda: SGD(lr=0.05), lambda p: torch.optim.SGD(p, lr=0.05)),
    (lambda: Momentum(lr=0.05, mu=0.9), lambda p: torch.optim.SGD(p, lr=0.05, momentum=0.9)),
    (
        lambda: RMSProp(lr=0.01, rho=0.9, eps=1e-8),
        lambda p: torch.optim.RMSprop(p, lr=0.01, alpha=0.9, eps=1e-8),
    ),
    (
        lambda: Adam(lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8),
        lambda p: torch.optim.Adam(p, lr=0.02, betas=(0.9, 0.999), eps=1e-8),
    ),
]


@pytest.mark.parametrize("make_ours, make_torch", PAIRS, ids=["sgd", "momentum", "rmsprop", "adam"])
def test_update_rules_match_pytorch_step_for_step(make_ours, make_torch):
    """Run 50 steps on a non-convex function and compare the whole trajectory."""

    def f_torch(x):
        return (x**4).sum() - 3 * (x**2).sum() + torch.sin(3 * x).sum()

    def grad_np(x):
        return 4 * x**3 - 6 * x + 3 * np.cos(3 * x)

    x0 = np.array([1.5, -0.7, 0.2])
    ours = make_ours()
    x = x0.copy()
    xt = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    opt = make_torch([xt])
    for _ in range(50):
        x = ours.step(x, grad_np(x))
        opt.zero_grad()
        f_torch(xt).backward()
        opt.step()
        np.testing.assert_allclose(x, xt.detach().numpy(), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    "optimizer, steps",
    [
        (SGD(lr=0.15), 300),
        (Momentum(lr=0.02, mu=0.9), 400),
        (RMSProp(lr=0.01, rho=0.9), 2000),
        (Adam(lr=0.05), 2000),
    ],
    ids=["sgd", "momentum", "rmsprop", "adam"],
)
def test_all_optimizers_minimise_a_quadratic(optimizer, steps):
    x, path = minimize(quad_grad, np.array([5.0, 5.0]), optimizer, steps=steps)
    np.testing.assert_allclose(x, X_STAR, atol=2e-2)
    assert quad_loss(path[-1]) < quad_loss(path[0])


def test_gd_diverges_when_learning_rate_exceeds_2_over_L():
    # Largest curvature L = 10, so the stability limit is lr < 2 / L = 0.2.
    x_ok, _ = minimize(quad_grad, np.array([1.0, 1.0]), SGD(lr=0.19), steps=500)
    x_bad, _ = minimize(quad_grad, np.array([1.0, 1.0]), SGD(lr=0.21), steps=500)
    assert np.linalg.norm(x_ok - X_STAR) < 1e-3
    assert np.linalg.norm(x_bad - X_STAR) > 1e3


def test_momentum_beats_plain_gd_on_ill_conditioned_problem():
    x0 = np.array([5.0, 5.0])
    plain, _ = minimize(quad_grad, x0, SGD(lr=0.02), steps=100)
    heavy, _ = minimize(quad_grad, x0, Momentum(lr=0.02, mu=0.9), steps=100)
    assert quad_loss(heavy) < quad_loss(plain)


def test_adam_first_step_has_size_lr_whatever_the_gradient_scale():
    # Bias correction makes m_hat / sqrt(v_hat) = g / |g| = sign(g) on step one.
    for scale in (1e-3, 1.0, 1e6):
        g = scale * np.array([1.0, -2.0, 3.0])
        x = Adam(lr=0.1).step(np.zeros(3), g)
        np.testing.assert_allclose(x, -0.1 * np.sign(g), rtol=1e-4)


def test_minibatch_sgd_fits_linear_regression(rng):
    n, d = 500, 3
    X = rng.normal(size=(n, d))
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true + 0.01 * rng.normal(size=n)

    def grad(w, idx):  # gradient of mean squared error on the mini-batch
        return 2 * X[idx].T @ (X[idx] @ w - y[idx]) / len(idx)

    w_closed_form = np.linalg.lstsq(X, y, rcond=None)[0]
    w_sgd = minibatch_sgd(grad, np.zeros(d), n, SGD(lr=0.05), batch_size=32, epochs=40)
    w_batch = minibatch_sgd(grad, np.zeros(d), n, SGD(lr=0.1), batch_size=n, epochs=300)
    np.testing.assert_allclose(w_batch, w_closed_form, atol=1e-6)  # full batch: exact
    np.testing.assert_allclose(w_sgd, w_closed_form, atol=2e-2)  # mini-batch: noisy but close
