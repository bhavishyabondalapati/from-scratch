import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as SkPCA
from sklearn.linear_model import LogisticRegression

from .impl import PCA


@pytest.fixture(scope="module")
def digits():
    return load_digits(return_X_y=True)


def align(a, b):
    return a * np.sign(np.sum(a * b, axis=0, keepdims=True))


@pytest.mark.parametrize("whiten", [False, True])
def test_matches_sklearn(digits, whiten):
    X, _ = digits
    ours, ref = PCA(10, whiten=whiten).fit(X), SkPCA(10, whiten=whiten).fit(X)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-8)
    np.testing.assert_allclose(
        ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-8
    )
    Z, Zr = ours.transform(X), ref.transform(X)
    np.testing.assert_allclose(align(Z, Zr), Zr, atol=1e-7)
    np.testing.assert_allclose(ours.inverse_transform(Z), ref.inverse_transform(Zr), atol=1e-6)
    if whiten:
        np.testing.assert_allclose(Z.var(axis=0, ddof=1), 1, rtol=1e-8)


def test_variance_fraction_picks_the_same_k_as_sklearn(digits):
    X, _ = digits
    for frac in (0.5, 0.8, 0.95):
        ours, ref = PCA(frac).fit(X), SkPCA(frac).fit(X)
        assert ours.n_components_ == ref.n_components_
        assert ours.explained_variance_ratio_.sum() >= frac


def test_compression_keeps_most_of_the_signal(digits):
    X, y = digits
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[:1200], idx[1200:]
    pca = PCA(20).fit(X[tr])  # 64 pixels -> 20 numbers
    clf = LogisticRegression(max_iter=2000).fit(pca.transform(X[tr]), y[tr])
    acc = clf.score(pca.transform(X[te]), y[te])
    assert acc > 0.9 > np.bincount(y[te]).max() / len(te)
    error = np.mean((pca.inverse_transform(pca.transform(X[te])) - X[te]) ** 2)
    assert error < 0.25 * np.mean((X[te] - pca.mean_) ** 2)
