import numpy as np
import pytest
import torch

from .impl import (
    check_gradient,
    forward_difference_gradient,
    numerical_gradient,
    numerical_jacobian,
    relative_error,
)


def test_matches_known_analytic_gradients(rng):
    x = rng.normal(size=(3, 4))
    cases = [
        (lambda z: np.sum(z**2), 2 * x),
        (lambda z: np.sum(np.sin(z)), np.cos(x)),
        (lambda z: np.sum(np.exp(z)) / 3, np.exp(x) / 3),
        (lambda z: np.log(np.sum(np.exp(z))), np.exp(x) / np.exp(x).sum()),  # softmax
    ]
    for f, analytic in cases:
        ok, err = check_gradient(f, analytic, x)
        assert ok and err < 1e-7


def test_matches_torch_autograd_on_a_composite_function(rng):
    W, x = rng.normal(size=(4, 3)), rng.normal(size=3)

    def f(w):
        return float(np.sum(np.tanh(w @ x) ** 2))

    Wt = torch.tensor(W, requires_grad=True)
    (torch.tanh(Wt @ torch.tensor(x)) ** 2).sum().backward()
    assert relative_error(numerical_gradient(f, W), Wt.grad.numpy()) < 1e-7


def test_detects_a_wrong_gradient(rng):
    x = rng.normal(size=5)
    ok, err = check_gradient(lambda z: np.sum(z**3), 3 * x**2, x)
    assert ok
    wrong = 3 * x**2
    wrong[2] *= 1.01  # a 1% bug in a single coordinate must be caught
    ok, err = check_gradient(lambda z: np.sum(z**3), wrong, x)
    assert not ok and err > 1e-3
    ok, _ = check_gradient(lambda z: np.sum(z**3), 2 * x**2, x)  # wrong constant
    assert not ok


def test_input_is_restored_and_int_arrays_rejected(rng):
    x = rng.normal(size=6)
    before = x.copy()
    numerical_gradient(lambda z: np.sum(z**2), x)
    np.testing.assert_array_equal(x, before)
    with pytest.raises(TypeError):
        numerical_gradient(lambda z: np.sum(z**2), np.array([1, 2, 3]))


def test_central_difference_is_more_accurate_than_forward(rng):
    x = rng.normal(size=8)
    f, exact = (lambda z: np.sum(np.exp(z))), np.exp(x)
    central = np.abs(numerical_gradient(f, x, eps=1e-4) - exact).max()
    forward = np.abs(forward_difference_gradient(f, x, eps=1e-4) - exact).max()
    assert central < forward / 100  # O(eps^2) vs O(eps)


def test_error_is_u_shaped_in_eps():
    # Too large: truncation error. Too small: floating-point cancellation.
    x = np.array([1.0])
    errors = {
        eps: abs(numerical_gradient(lambda z: np.sin(z[0]), x, eps=eps)[0] - np.cos(1.0))
        for eps in (1e-1, 1e-5, 1e-13)
    }
    assert errors[1e-5] < errors[1e-1] and errors[1e-5] < errors[1e-13]


def test_relative_error_is_scale_free():
    a = np.array([1.0, 2.0])
    assert relative_error(a, a) == 0
    assert relative_error(1e6 * a, 1e6 * a * 1.001) == pytest.approx(relative_error(a, a * 1.001))


def test_jacobian_matches_torch(rng):
    x = rng.normal(size=4)
    W = rng.normal(size=(3, 4))
    J = numerical_jacobian(lambda z: np.tanh(W @ z), x)
    Wt = torch.tensor(W)
    Jt = torch.autograd.functional.jacobian(lambda z: torch.tanh(Wt @ z), torch.tensor(x))
    np.testing.assert_allclose(J, Jt.numpy(), atol=1e-8)
    # Linear map: the Jacobian is the matrix itself.
    np.testing.assert_allclose(numerical_jacobian(lambda z: W @ z, x), W, atol=1e-8)
