import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.preprocessing import StandardScaler

from s2_math.gradient_check.impl import check_gradient

from .impl import (
    LinearRegression,
    add_bias_column,
    mse,
    mse_gradient,
    normal_equations,
    r2_score,
)


@pytest.fixture(scope="module")
def diabetes():
    X, y = load_diabetes(return_X_y=True)
    return StandardScaler().fit_transform(X), y


def test_closed_form_matches_sklearn(diabetes):
    X, y = diabetes
    ours, ref = LinearRegression().fit(X, y), SkLinearRegression().fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-6, atol=1e-6)
    assert ours.intercept_ == pytest.approx(ref.intercept_)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-8)
    assert ours.score(X, y) == pytest.approx(ref.score(X, y))


def test_normal_equations_agree_with_lstsq(rng):
    X = add_bias_column(rng.normal(size=(100, 4)))
    y = rng.normal(size=100)
    np.testing.assert_allclose(normal_equations(X, y), np.linalg.lstsq(X, y, rcond=None)[0])


def test_gradient_descent_converges_to_closed_form(rng):
    X = rng.normal(size=(300, 5))
    y = X @ np.array([1.5, -2.0, 0.0, 0.7, 3.0]) + 4.0 + 0.1 * rng.normal(size=300)
    exact = LinearRegression().fit(X, y)
    gd = LinearRegression(method="gd", lr=0.05, n_iters=3000).fit(X, y)
    np.testing.assert_allclose(gd.coef_, exact.coef_, atol=1e-6)
    assert gd.intercept_ == pytest.approx(exact.intercept_, abs=1e-6)
    losses = gd.loss_history_
    assert all(a >= b - 1e-12 for a, b in zip(losses, losses[1:]))  # loss never goes up


def test_mse_gradient_passes_finite_difference_check(rng):
    Xb = add_bias_column(rng.normal(size=(20, 3)))
    y, w = rng.normal(size=20), rng.normal(size=4)
    ok, err = check_gradient(lambda v: mse(y, Xb @ v), mse_gradient(Xb, y, w), w)
    assert ok, err


def test_recovers_exact_coefficients_without_noise(rng):
    X = rng.normal(size=(50, 3))
    model = LinearRegression().fit(X, X @ np.array([2.0, -1.0, 0.5]) + 7.0)
    np.testing.assert_allclose(model.coef_, [2.0, -1.0, 0.5], atol=1e-10)
    assert model.intercept_ == pytest.approx(7.0)


def test_handles_perfectly_collinear_features(rng):
    x = rng.normal(size=(40, 1))
    X = np.hstack([x, 2 * x])  # X^T X is singular: the inverse does not exist
    y = 3 * x[:, 0] + 1
    model = LinearRegression().fit(X, y)
    np.testing.assert_allclose(model.predict(X), y, atol=1e-8)


def test_beats_mean_baseline_on_held_out_data(diabetes):
    X, y = diabetes
    train, test = slice(0, 300), slice(300, None)
    model = LinearRegression().fit(X[train], y[train])
    baseline = mse(y[test], np.full(len(y[test]), y[train].mean()))
    assert mse(y[test], model.predict(X[test])) < 0.6 * baseline
    assert r2_score(y[test], model.predict(X[test])) > 0.4
    assert r2_score(y, np.full(len(y), y.mean())) == pytest.approx(0.0)


def test_invalid_method():
    with pytest.raises(ValueError):
        LinearRegression(method="magic")
