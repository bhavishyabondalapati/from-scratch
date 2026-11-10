import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.linear_model import LogisticRegression as SkLogReg
from sklearn.preprocessing import StandardScaler

from s2_math.gradient_check.impl import check_gradient

from .impl import (
    LogisticRegression,
    SoftmaxRegression,
    binary_loss_and_grad,
    softmax_loss_and_grad,
)


@pytest.fixture(scope="module")
def cancer():
    X, y = load_breast_cancer(return_X_y=True)
    return StandardScaler().fit_transform(X[:, :10]), y


@pytest.fixture(scope="module")
def iris():
    X, y = load_iris(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def test_binary_gradient_check(rng):
    X, y = rng.normal(size=(30, 4)), rng.integers(0, 2, size=30).astype(float)
    w, b = rng.normal(size=4), 0.3
    _, grad_w, grad_b = binary_loss_and_grad(w, b, X, y, l2=0.1)
    ok, err = check_gradient(lambda v: binary_loss_and_grad(v, b, X, y, 0.1)[0], grad_w, w)
    assert ok, err
    bb = np.array([b])
    ok, err = check_gradient(
        lambda v: binary_loss_and_grad(w, v[0], X, y, 0.1)[0], np.array([grad_b]), bb
    )
    assert ok, err


def test_softmax_gradient_check(rng):
    X, y = rng.normal(size=(30, 4)), rng.integers(0, 3, size=30)
    W, b = rng.normal(size=(4, 3)), rng.normal(size=3)
    _, grad_W, grad_b = softmax_loss_and_grad(W, b, X, y, l2=0.1)
    ok, err = check_gradient(lambda v: softmax_loss_and_grad(v, b, X, y, 0.1)[0], grad_W, W)
    assert ok, err
    ok, err = check_gradient(lambda v: softmax_loss_and_grad(W, v, X, y, 0.1)[0], grad_b, b)
    assert ok, err


def test_binary_matches_sklearn(cancer):
    X, y = cancer
    l2 = 0.01
    ours = LogisticRegression(lr=0.5, n_iters=8000, l2=l2).fit(X, y)
    # sklearn minimises 0.5 ||w||^2 + C * sum(loss); dividing by C * n gives our objective.
    ref = SkLogReg(C=1 / (l2 * len(y)), tol=1e-10, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_[0], atol=2e-3)
    assert ours.intercept_ == pytest.approx(ref.intercept_[0], abs=2e-3)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X)[:, 1], atol=1e-3)
    assert (ours.predict(X) == ref.predict(X)).all()


def test_multinomial_matches_sklearn(iris):
    X, y = iris
    l2 = 0.01
    ours = SoftmaxRegression(lr=0.5, n_iters=8000, l2=l2).fit(X, y)
    ref = SkLogReg(C=1 / (l2 * len(y)), tol=1e-10, max_iter=10_000).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-3)
    # The L2 penalty makes the softmax weights unique, so they can be compared directly.
    np.testing.assert_allclose(ours.coef_.T, ref.coef_, atol=5e-3)


def test_loss_decreases_and_beats_majority_baseline(cancer):
    X, y = cancer
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    train, test = idx[:400], idx[400:]
    model = LogisticRegression(n_iters=2000, l2=0.001).fit(X[train], y[train])
    losses = model.loss_history_
    assert all(a >= b - 1e-12 for a, b in zip(losses, losses[1:]))
    accuracy = np.mean(model.predict(X[test]) == y[test])
    majority = max(np.mean(y[test]), 1 - np.mean(y[test]))
    assert accuracy > 0.9 > majority


def test_softmax_with_two_classes_equals_binary_model(cancer):
    X, y = cancer
    binary = LogisticRegression(lr=0.5, n_iters=3000).fit(X, y)
    multi = SoftmaxRegression(lr=0.5, n_iters=3000).fit(X, y)
    # sigmoid(z) == softmax([0, z])[1]; with zero init the two columns stay mirror images,
    # so the softmax model's logit difference is the binary model's logit (lr is halved per column).
    np.testing.assert_allclose(multi.coef_[:, 1], -multi.coef_[:, 0], atol=1e-10)
    assert np.mean(multi.predict(X) == binary.predict(X)) > 0.99


def test_probabilities_are_valid_and_threshold_moves_predictions(iris, cancer):
    X, y = iris
    P = SoftmaxRegression(n_iters=500).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    assert ((P >= 0) & (P <= 1)).all()
    Xc, yc = cancer
    model = LogisticRegression(n_iters=500).fit(Xc, yc)
    assert model.predict(Xc, threshold=0.9).sum() < model.predict(Xc, threshold=0.1).sum()
