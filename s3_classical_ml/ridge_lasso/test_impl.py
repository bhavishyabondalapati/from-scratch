import numpy as np
import pytest
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso as SkLasso
from sklearn.linear_model import Ridge as SkRidge
from sklearn.preprocessing import StandardScaler

from .impl import Lasso, Ridge, lasso_objective, ridge_objective, soft_threshold


@pytest.fixture(scope="module")
def data():
    X, y = load_diabetes(return_X_y=True)
    return StandardScaler().fit_transform(X), y


@pytest.mark.parametrize("alpha", [0.01, 1.0, 100.0])
def test_ridge_matches_sklearn(data, alpha):
    X, y = data
    ours, ref = Ridge(alpha).fit(X, y), SkRidge(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, rtol=1e-7, atol=1e-8)
    assert ours.intercept_ == pytest.approx(ref.intercept_)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 5.0, 20.0])
def test_lasso_matches_sklearn(data, alpha):
    X, y = data
    ours = Lasso(alpha, max_iter=10_000, tol=1e-10).fit(X, y)
    ref = SkLasso(alpha=alpha, max_iter=100_000, tol=1e-12).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_, atol=1e-5)
    assert ours.intercept_ == pytest.approx(ref.intercept_)
    assert set(np.flatnonzero(ours.coef_)) == set(np.flatnonzero(ref.coef_))  # same sparsity


def test_solutions_are_local_minima_of_their_objectives(data, rng):
    X, y = data
    ridge, lasso = Ridge(10.0).fit(X, y), Lasso(2.0, tol=1e-12).fit(X, y)
    best_r = ridge_objective(X, y, ridge.coef_, ridge.intercept_, 10.0)
    best_l = lasso_objective(X, y, lasso.coef_, lasso.intercept_, 2.0)
    for _ in range(200):  # no small random nudge may improve a convex optimum
        dw, db = rng.normal(size=X.shape[1]) * 0.05, rng.normal() * 0.05
        assert ridge_objective(X, y, ridge.coef_ + dw, ridge.intercept_ + db, 10.0) >= best_r
        assert lasso_objective(X, y, lasso.coef_ + dw, lasso.intercept_ + db, 2.0) >= best_l - 1e-9


def test_soft_threshold():
    assert soft_threshold(3.0, 1.0) == 2.0
    assert soft_threshold(-3.0, 1.0) == -2.0
    assert soft_threshold(0.5, 1.0) == 0.0 and soft_threshold(-1.0, 1.0) == 0.0


def test_ridge_shrinks_and_lasso_zeroes(data):
    X, y = data
    ridge_norms = [np.linalg.norm(Ridge(a).fit(X, y).coef_) for a in (0.0, 10.0, 1000.0, 1e6)]
    assert all(a > b for a, b in zip(ridge_norms, ridge_norms[1:]))  # monotone shrinkage
    assert np.all(Ridge(1000.0).fit(X, y).coef_ != 0)  # ...but never exactly zero

    n_zero = [int(np.sum(Lasso(a).fit(X, y).coef_ == 0)) for a in (0.01, 1.0, 10.0, 100.0)]
    assert n_zero == sorted(n_zero) and n_zero[0] < n_zero[-1]
    # Above alpha_max = max |X^T y| / n every coefficient is exactly zero.
    alpha_max = np.max(np.abs((X - X.mean(0)).T @ (y - y.mean()))) / len(y)
    assert np.all(Lasso(alpha_max * 1.001).fit(X, y).coef_ == 0)
    assert np.any(Lasso(alpha_max * 0.99).fit(X, y).coef_ != 0)


def test_alpha_zero_ridge_is_ordinary_least_squares(data):
    X, y = data
    ols = np.linalg.lstsq(np.hstack([X, np.ones((len(X), 1))]), y, rcond=None)[0]
    np.testing.assert_allclose(Ridge(0.0).fit(X, y).coef_, ols[:-1], atol=1e-6)


def test_lasso_recovers_sparse_truth_and_ignores_noise_features(rng):
    X = rng.normal(size=(200, 30))
    true_w = np.zeros(30)
    true_w[[2, 11, 25]] = [4.0, -3.0, 2.0]
    y = X @ true_w + 0.1 * rng.normal(size=200)
    model = Lasso(alpha=0.1).fit(X, y)
    assert set(np.flatnonzero(model.coef_)) == {2, 11, 25}
    np.testing.assert_allclose(model.coef_[[2, 11, 25]], [4.0, -3.0, 2.0], atol=0.2)


def test_ridge_beats_least_squares_when_samples_are_scarce(rng):
    # With n barely above d, least squares fits the noise and its weights blow up.
    n, d = 55, 50
    w = rng.normal(size=d)
    X, X_test = rng.normal(size=(n, d)), rng.normal(size=(2000, d))
    y = X @ w + 2.0 * rng.normal(size=n)
    y_test = X_test @ w

    def test_error(alpha):
        return np.mean((Ridge(alpha).fit(X, y).predict(X_test) - y_test) ** 2)

    assert test_error(10.0) < 0.5 * test_error(1e-8)
