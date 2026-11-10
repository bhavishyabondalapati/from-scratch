import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

from .impl import explained_variance_ratio, inverse_transform, pca_eig, pca_svd, transform


def align(a, b):
    """Flip rows of a so they point the same way as rows of b (sign is arbitrary)."""
    return a * np.sign(np.sum(a * b, axis=1, keepdims=True))


@pytest.mark.parametrize("pca", [pca_eig, pca_svd])
def test_matches_sklearn_on_iris(pca):
    X = load_iris().data
    k = 3
    ref = PCA(n_components=k).fit(X)
    components, variance, mean = pca(X, k)
    np.testing.assert_allclose(align(components, ref.components_), ref.components_, atol=1e-8)
    np.testing.assert_allclose(variance, ref.explained_variance_, rtol=1e-8)
    np.testing.assert_allclose(mean, ref.mean_)
    np.testing.assert_allclose(
        explained_variance_ratio(X, variance), ref.explained_variance_ratio_, rtol=1e-8
    )
    Z, Z_ref = transform(X, components, mean), ref.transform(X)
    np.testing.assert_allclose(np.abs(Z), np.abs(Z_ref), atol=1e-8)


def test_eig_and_svd_agree_exactly(rng):
    for _ in range(20):
        n, d = rng.integers(5, 40), rng.integers(2, 7)
        X = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))  # correlated features
        k = rng.integers(1, min(n - 1, d) + 1)
        c1, v1, m1 = pca_eig(X, k)
        c2, v2, m2 = pca_svd(X, k)
        np.testing.assert_allclose(c1, c2, atol=1e-7)  # identical thanks to the sign convention
        np.testing.assert_allclose(v1, v2, atol=1e-9)
        np.testing.assert_allclose(m1, m2)


def test_components_are_orthonormal_and_scores_uncorrelated(rng):
    X = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
    components, variance, mean = pca_svd(X, 5)
    np.testing.assert_allclose(components @ components.T, np.eye(5), atol=1e-10)
    Z = transform(X, components, mean)
    # In the new basis the covariance is diagonal, with the eigenvalues on the diagonal.
    np.testing.assert_allclose(np.cov(Z.T), np.diag(variance), atol=1e-8)
    assert np.all(np.diff(variance) <= 1e-12)  # sorted, largest first


def test_reconstruction(rng):
    X = rng.normal(size=(100, 6)) @ rng.normal(size=(6, 6))
    c, v, m = pca_svd(X, 6)
    np.testing.assert_allclose(
        inverse_transform(transform(X, c, m), c, m), X, atol=1e-10
    )  # k = d: lossless
    errors = []
    for k in range(1, 7):
        c, v, m = pca_svd(X, k)
        errors.append(np.sum((X - inverse_transform(transform(X, c, m), c, m)) ** 2))
    assert all(a >= b - 1e-9 for a, b in zip(errors, errors[1:]))  # more components, less error
    # The squared error equals the variance thrown away: (n - 1) * sum of dropped eigenvalues.
    _, all_var, _ = pca_svd(X, 6)
    assert errors[1] == pytest.approx((len(X) - 1) * all_var[2:].sum())


def test_recovers_a_planted_direction(rng):
    direction = np.array([3.0, 4.0]) / 5
    t = rng.normal(size=500) * 10
    X = np.outer(t, direction) + rng.normal(size=(500, 2)) * 0.1 + np.array([100.0, -50.0])
    components, variance, _ = pca_eig(X, 2)
    assert abs(components[0] @ direction) > 0.999
    assert variance[0] / variance.sum() > 0.99


def test_pca_beats_random_projection(rng):
    X = load_iris().data
    c, _, m = pca_svd(X, 2)
    pca_error = np.sum((X - inverse_transform(transform(X, c, m), c, m)) ** 2)
    Q, _ = np.linalg.qr(rng.normal(size=(4, 2)))  # a random orthonormal 2D basis
    random_error = np.sum((X - inverse_transform(transform(X, Q.T, m), Q.T, m)) ** 2)
    assert pca_error < random_error
