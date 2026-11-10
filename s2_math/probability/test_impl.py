import numpy as np
import pytest
from scipy import optimize, stats

from .impl import (
    bayes_posterior,
    bayes_update,
    bernoulli_log_likelihood,
    bernoulli_mle,
    beta_map,
    beta_mean,
    beta_posterior,
    categorical_mle,
    exponential_mle,
    gaussian_log_likelihood,
    gaussian_mle,
    poisson_log_likelihood,
    poisson_mle,
)


def test_bayes_medical_test_example():
    # 1% prevalence, 99% sensitivity, 5% false positives -> a positive test means ~16.7%.
    assert bayes_posterior(0.01, 0.99, 0.05) == pytest.approx(0.0099 / (0.0099 + 0.0495))
    assert bayes_posterior(0.5, 0.9, 0.9) == pytest.approx(0.5)  # uninformative test


def test_bayes_matches_simulation(rng):
    prior, sens, fpr = 0.2, 0.9, 0.1
    sick = rng.random(400_000) < prior
    positive = np.where(sick, rng.random(400_000) < sens, rng.random(400_000) < fpr)
    assert sick[positive].mean() == pytest.approx(bayes_posterior(prior, sens, fpr), abs=5e-3)


def test_bayes_update_multiple_hypotheses():
    # Three coins with P(heads) = 0.2, 0.5, 0.8, picked uniformly; we observe one head.
    post = bayes_update([1 / 3, 1 / 3, 1 / 3], [0.2, 0.5, 0.8])
    np.testing.assert_allclose(post, [0.2 / 1.5, 0.5 / 1.5, 0.8 / 1.5])
    # Updating twice equals one update with the product of likelihoods.
    twice = bayes_update(bayes_update([0.5, 0.5], [0.3, 0.6]), [0.9, 0.2])
    np.testing.assert_allclose(twice, bayes_update([0.5, 0.5], [0.27, 0.12]))


def test_closed_form_mles_match_scipy_fit(rng):
    x = rng.normal(3.0, 2.0, size=500)
    mu, var = gaussian_mle(x)
    mu_ref, sigma_ref = stats.norm.fit(x)
    assert mu == pytest.approx(mu_ref) and var == pytest.approx(sigma_ref**2)
    assert var == pytest.approx(np.var(x))  # divisor n, not n - 1

    waits = rng.exponential(scale=0.5, size=500)
    _, scale_ref = stats.expon.fit(waits, floc=0)
    assert exponential_mle(waits) == pytest.approx(1 / scale_ref)


def maximise(neg_log_lik, bounds):
    return optimize.minimize_scalar(neg_log_lik, bounds=bounds, method="bounded").x


def test_closed_forms_really_maximise_the_likelihood(rng):
    flips = (rng.random(300) < 0.3).astype(float)
    p_hat = maximise(lambda p: -bernoulli_log_likelihood(flips, p), (1e-6, 1 - 1e-6))
    assert bernoulli_mle(flips) == pytest.approx(p_hat, abs=1e-4)

    counts = rng.poisson(4.2, size=300)
    lam_hat = maximise(lambda lam: -poisson_log_likelihood(counts, lam), (1e-6, 50))
    assert poisson_mle(counts) == pytest.approx(lam_hat, abs=1e-4)

    x = rng.normal(-1.0, 3.0, size=300)
    mu, var = gaussian_mle(x)
    best = optimize.minimize(
        lambda t: -gaussian_log_likelihood(x, t[0], np.exp(t[1])), x0=[0.0, 0.0]
    ).x
    assert mu == pytest.approx(best[0], abs=1e-3)
    assert var == pytest.approx(np.exp(best[1]), rel=1e-3)


def test_log_likelihoods_match_scipy(rng):
    x = rng.normal(size=50)
    assert gaussian_log_likelihood(x, 0.3, 1.7) == pytest.approx(
        stats.norm.logpdf(x, 0.3, np.sqrt(1.7)).sum()
    )
    k = rng.poisson(3, size=50)
    assert poisson_log_likelihood(k, 2.5) == pytest.approx(stats.poisson.logpmf(k, 2.5).sum())
    b = rng.integers(0, 2, size=50)
    assert bernoulli_log_likelihood(b, 0.4) == pytest.approx(stats.bernoulli.logpmf(b, 0.4).sum())


def test_mle_is_consistent(rng):
    # More data -> estimate closer to the truth.
    errors = [abs(bernoulli_mle(rng.random(n) < 0.37) - 0.37) for n in (100, 1_000_000)]
    assert errors[1] < errors[0] and errors[1] < 2e-3


def test_categorical_mle_and_smoothing():
    labels = np.array([0, 0, 1, 0, 3])
    np.testing.assert_allclose(categorical_mle(labels, 4), [0.6, 0.2, 0.0, 0.2])
    smoothed = categorical_mle(labels, 4, alpha=1.0)
    np.testing.assert_allclose(smoothed, [4 / 9, 2 / 9, 1 / 9, 2 / 9])
    assert smoothed.min() > 0 and smoothed.sum() == pytest.approx(1)


def test_beta_posterior_against_scipy_and_limits():
    a, b = beta_posterior(heads=7, tails=3, a=2, b=2)
    assert (a, b) == (9, 5)
    assert beta_mean(a, b) == pytest.approx(stats.beta(a, b).mean())
    # MAP is where the posterior density peaks.
    grid = np.linspace(0.001, 0.999, 9999)
    assert beta_map(a, b) == pytest.approx(grid[np.argmax(stats.beta(a, b).pdf(grid))], abs=1e-3)
    # Uniform prior: MAP = MLE.
    assert beta_map(*beta_posterior(7, 3)) == pytest.approx(0.7)
    # Zero heads in 3 flips: MLE says "heads is impossible", the posterior mean does not.
    assert bernoulli_mle(np.zeros(3)) == 0.0
    assert beta_mean(*beta_posterior(0, 3)) == pytest.approx(0.2)
    # With lots of data the prior stops mattering.
    assert beta_mean(*beta_posterior(7000, 3000, a=50, b=1)) == pytest.approx(0.7, abs=5e-3)
