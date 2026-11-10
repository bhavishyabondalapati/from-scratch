import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE as SkTSNE
from sklearn.manifold import trustworthiness

from s2_math.gradient_check.impl import check_gradient
from s3_classical_ml.knn.impl import pairwise_sq_distances

from .impl import TSNE, conditional_probabilities, kl_divergence_and_grad


@pytest.fixture(scope="module")
def digits_small():
    X, y = load_digits(return_X_y=True)
    idx = np.random.default_rng(0).permutation(len(y))[:400]
    return X[idx] / 16.0, y[idx]


@pytest.fixture(scope="module")
def embedding(digits_small):
    X, _ = digits_small
    return TSNE(perplexity=30, n_iter=600, random_state=0).fit_transform(X)


def test_binary_search_hits_target_perplexity(rng):
    X = rng.normal(size=(120, 5))
    for perp in (5.0, 20.0, 50.0):
        P = conditional_probabilities(pairwise_sq_distances(X, X), perp)
        np.testing.assert_allclose(P.sum(axis=1), 1)
        assert np.allclose(np.diag(P), 0)
        H = -np.sum(np.where(P > 0, P * np.log(np.where(P > 0, P, 1)), 0), axis=1)
        np.testing.assert_allclose(np.exp(H), perp, rtol=1e-3)


def test_kl_gradient_passes_finite_difference_check(rng):
    n = 12
    P = rng.random((n, n))
    np.fill_diagonal(P, 0)
    P = (P + P.T) / (P + P.T).sum()
    Y = rng.normal(size=(n, 2))
    _, grad = kl_divergence_and_grad(P, Y)
    ok, err = check_gradient(lambda y: kl_divergence_and_grad(P, y)[0], grad, Y)
    assert ok, err


def test_embedding_preserves_neighbourhoods_like_sklearn(digits_small, embedding):
    X, _ = digits_small
    ours = trustworthiness(X, embedding, n_neighbors=10)
    ref_emb = SkTSNE(perplexity=30, method="exact", init="random", random_state=0).fit_transform(X)
    ref = trustworthiness(X, ref_emb, n_neighbors=10)
    assert ours > 0.95
    assert ours == pytest.approx(ref, abs=0.02)


def test_classes_separate_in_2d(digits_small, embedding):
    _, y = digits_small
    # Leave-one-out 1-NN accuracy in the 2D map vs a random 2D projection.
    d = pairwise_sq_distances(embedding, embedding)
    np.fill_diagonal(d, np.inf)
    acc = np.mean(y[d.argmin(axis=1)] == y)
    rand = digits_small[0] @ np.random.default_rng(0).normal(size=(64, 2))
    dr = pairwise_sq_distances(rand, rand)
    np.fill_diagonal(dr, np.inf)
    assert acc > 0.9 > np.mean(y[dr.argmin(axis=1)] == y)


def test_kl_decreases_after_exaggeration(digits_small):
    X, _ = digits_small
    model = TSNE(n_iter=500, exaggeration_iters=100, random_state=1)
    model.fit_transform(X[:150])
    h = model.kl_history_
    assert h[-1] < h[0]
    assert model.kl_divergence_ == pytest.approx(h[-1], rel=0.05)
