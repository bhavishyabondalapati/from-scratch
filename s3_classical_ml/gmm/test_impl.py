import numpy as np
import pytest
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture as SkGMM

from .impl import GaussianMixture, gaussian_log_pdf


def make_mixture(rng, n=900):
    means = np.array([[0.0, 0.0], [6.0, 1.0], [2.0, 7.0]])
    covs = np.array(
        [[[1.0, 0.8], [0.8, 1.5]], [[0.5, 0.0], [0.0, 2.0]], [[2.0, -1.0], [-1.0, 1.0]]]
    )
    weights = np.array([0.5, 0.3, 0.2])
    z = rng.choice(3, size=n, p=weights)
    X = np.array([rng.multivariate_normal(means[k], covs[k]) for k in z])
    return X, z, means, covs, weights


def test_log_pdf_matches_scipy(rng):
    A = rng.normal(size=(3, 3))
    cov, mean, X = A @ A.T + np.eye(3), rng.normal(size=3), rng.normal(size=(20, 3))
    np.testing.assert_allclose(
        gaussian_log_pdf(X, mean, cov), multivariate_normal(mean, cov).logpdf(X)
    )


def test_recovers_true_parameters(rng):
    X, z, means, covs, weights = make_mixture(rng)
    gmm = GaussianMixture(3, random_state=0).fit(X)
    order = [int(np.argmin(((gmm.means_ - m) ** 2).sum(1))) for m in means]  # match components
    assert sorted(order) == [0, 1, 2]
    np.testing.assert_allclose(gmm.means_[order], means, atol=0.25)
    np.testing.assert_allclose(gmm.covariances_[order], covs, atol=0.4)
    np.testing.assert_allclose(gmm.weights_[order], weights, atol=0.05)
    assert adjusted_rand_score(z, gmm.predict(X)) > 0.9


def test_matches_sklearn_on_iris():
    X = load_iris().data
    ours = GaussianMixture(3, tol=1e-10, max_iter=1000, n_init=10, random_state=0).fit(X)
    ref = SkGMM(3, covariance_type="full", tol=1e-10, max_iter=1000, n_init=10, random_state=0).fit(
        X
    )
    assert ours.score(X) == pytest.approx(ref.score(X), rel=1e-3)
    assert adjusted_rand_score(ref.predict(X), ours.predict(X)) > 0.95


def test_em_never_decreases_log_likelihood(rng):
    X, *_ = make_mixture(rng, n=400)
    gmm = GaussianMixture(4, random_state=1, tol=0).fit(X)
    h = gmm.log_likelihood_history_
    assert all(b >= a - 1e-9 for a, b in zip(h, h[1:]))


def test_responsibilities_and_fit_beat_single_gaussian(rng):
    X, *_ = make_mixture(rng)
    gmm = GaussianMixture(3, random_state=0).fit(X)
    P = gmm.predict_proba(X)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    one = GaussianMixture(1).fit(X)
    assert gmm.score(X) > one.score(X) + 0.3  # far better density model than one blob
    # BIC should prefer the true number of components.
    bics = {k: GaussianMixture(k, random_state=0).fit(X).bic(X) for k in (1, 2, 3, 6)}
    assert min(bics, key=bics.get) == 3


def test_sampling_reproduces_the_mixture(rng):
    X, *_ = make_mixture(rng)
    gmm = GaussianMixture(3, random_state=0).fit(X)
    S, comps = gmm.sample(4000, seed=0)
    np.testing.assert_allclose(S.mean(axis=0), X.mean(axis=0), atol=0.2)
    np.testing.assert_allclose(np.bincount(comps, minlength=3) / 4000, gmm.weights_, atol=0.03)
