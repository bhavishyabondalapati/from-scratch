import warnings

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy import special

from s2_math.gradient_check.impl import check_gradient

from .impl import (
    binary_cross_entropy_with_logits,
    cross_entropy,
    cross_entropy_grad,
    log_sigmoid,
    log_softmax,
    logsumexp,
    naive_softmax,
    sigmoid,
    softmax,
)

EXTREME = np.array([[1000.0, 1001.0, 999.0], [-1000.0, -1001.0, -999.0], [0.0, 1e4, -1e4]])


def test_match_scipy_on_ordinary_inputs(rng):
    z = rng.normal(size=(5, 7)) * 3
    for axis in (0, 1, -1):
        np.testing.assert_allclose(softmax(z, axis), special.softmax(z, axis=axis))
        np.testing.assert_allclose(log_softmax(z, axis), special.log_softmax(z, axis=axis))
        np.testing.assert_allclose(logsumexp(z, axis), special.logsumexp(z, axis=axis))
    assert logsumexp(z, axis=1, keepdims=True).shape == (5, 1)
    np.testing.assert_allclose(sigmoid(z), special.expit(z))
    np.testing.assert_allclose(log_sigmoid(z), special.log_expit(z))


def test_stable_versions_survive_extreme_inputs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any overflow warning fails the test
        np.testing.assert_allclose(softmax(EXTREME), special.softmax(EXTREME, axis=-1))
        np.testing.assert_allclose(logsumexp(EXTREME), special.logsumexp(EXTREME, axis=-1))
        np.testing.assert_allclose(log_softmax(EXTREME), special.log_softmax(EXTREME, axis=-1))
        x = np.array([-1e4, -745.0, -30.0, 0.0, 30.0, 745.0, 1e4])
        np.testing.assert_allclose(sigmoid(x), special.expit(x))
        np.testing.assert_allclose(log_sigmoid(x), special.log_expit(x))
    assert np.isfinite(log_softmax(EXTREME)).all()


def test_naive_softmax_breaks_where_stable_one_does_not():
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(naive_softmax(EXTREME[0])).any()
    assert np.isfinite(softmax(EXTREME[0])).all()
    # ...but they agree where the naive version is safe.
    z = np.array([0.5, -1.0, 2.0])
    np.testing.assert_allclose(softmax(z), naive_softmax(z))


def test_softmax_properties(rng):
    z = rng.normal(size=(4, 6))
    p = softmax(z)
    np.testing.assert_allclose(p.sum(axis=-1), 1)
    assert (p > 0).all()
    np.testing.assert_allclose(softmax(z + 123.4), p)  # shift invariance
    assert (p.argmax(axis=-1) == z.argmax(axis=-1)).all()  # order preserved
    # Temperature: dividing logits by a small T sharpens toward one-hot.
    assert softmax(z / 0.01).max(axis=-1).min() > 0.99


def test_logsumexp_edge_cases():
    assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf  # log(0)
    assert logsumexp(np.array([-np.inf, 2.0])) == pytest.approx(2.0)
    # A smooth maximum: max(z) <= logsumexp(z) <= max(z) + log(n)
    z = np.array([1.0, 5.0, 3.0])
    assert 5.0 <= logsumexp(z) <= 5.0 + np.log(3)


def test_cross_entropy_matches_torch_and_gradient_checks(rng):
    logits = rng.normal(size=(6, 5)) * 4
    targets = rng.integers(0, 5, size=6)
    expected = F.cross_entropy(torch.tensor(logits), torch.tensor(targets)).item()
    assert cross_entropy(logits, targets) == pytest.approx(expected)
    ok, err = check_gradient(
        lambda z: cross_entropy(z, targets), cross_entropy_grad(logits, targets), logits
    )
    assert ok, err
    # Extreme logits: confident and right -> ~0 loss; confident and wrong -> huge but finite.
    big = np.array([[1e4, 0.0, 0.0]])
    assert cross_entropy(big, np.array([0])) == pytest.approx(0.0)
    assert cross_entropy(big, np.array([1])) == pytest.approx(1e4)


def test_bce_with_logits_matches_torch(rng):
    x = np.concatenate([rng.normal(size=20) * 5, [-800.0, 800.0]])
    y = rng.integers(0, 2, size=22).astype(float)
    expected = F.binary_cross_entropy_with_logits(torch.tensor(x), torch.tensor(y)).item()
    assert binary_cross_entropy_with_logits(x, y) == pytest.approx(expected)
