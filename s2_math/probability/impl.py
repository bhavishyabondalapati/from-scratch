"""Basic probability for ML: maximum likelihood estimation and Bayes' rule."""

from __future__ import annotations

import math

import numpy as np

# --- Bayes' rule ---------------------------------------------------------------


def bayes_posterior(prior: float, likelihood: float, false_positive_rate: float) -> float:
    """P(H | evidence) for a yes/no hypothesis H.

        P(H | E) = P(E | H) P(H) / [ P(E | H) P(H) + P(E | not H) P(not H) ]

    Example: prior = disease prevalence, likelihood = test sensitivity,
    false_positive_rate = 1 - specificity.
    """
    evidence = likelihood * prior + false_positive_rate * (1 - prior)
    return likelihood * prior / evidence


def bayes_update(priors: np.ndarray, likelihoods: np.ndarray) -> np.ndarray:
    """Posterior over several hypotheses: multiply prior by likelihood, then normalise."""
    unnormalised = np.asarray(priors, dtype=float) * np.asarray(likelihoods, dtype=float)
    return unnormalised / unnormalised.sum()


# --- Maximum likelihood estimation ----------------------------------------------
# MLE: choose the parameter value under which the observed data is most probable.
# For these distributions, setting the derivative of the log-likelihood to zero
# gives a closed form.


def bernoulli_mle(x: np.ndarray) -> float:
    """Coin with unknown bias p. MLE: the fraction of ones."""
    return float(np.mean(x))


def gaussian_mle(x: np.ndarray) -> tuple[float, float]:
    """Normal(mu, sigma^2). MLE: sample mean and the variance with divisor n.

    (Dividing by n is the MLE but is biased low; dividing by n - 1 is unbiased.)
    """
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    return float(mu), float(np.mean((x - mu) ** 2))


def poisson_mle(x: np.ndarray) -> float:
    """Poisson(lambda) for counts. MLE: the sample mean."""
    return float(np.mean(x))


def exponential_mle(x: np.ndarray) -> float:
    """Exponential(rate) for waiting times. MLE: 1 / sample mean."""
    return float(1 / np.mean(x))


def categorical_mle(x: np.ndarray, n_classes: int, alpha: float = 0.0) -> np.ndarray:
    """Class probabilities from integer labels: normalised counts.

    alpha > 0 adds "pseudo-counts" (Laplace smoothing), which is the MAP
    estimate under a Dirichlet prior and keeps unseen classes from getting
    probability exactly 0.
    """
    counts = np.bincount(np.asarray(x), minlength=n_classes).astype(float) + alpha
    return counts / counts.sum()


# --- Log-likelihoods (what MLE maximises) ----------------------------------------


def bernoulli_log_likelihood(x: np.ndarray, p: float) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * np.log(p) + (1 - x) * np.log(1 - p)))


def gaussian_log_likelihood(x: np.ndarray, mu: float, var: float) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sum(-0.5 * np.log(2 * np.pi * var) - (x - mu) ** 2 / (2 * var)))


def poisson_log_likelihood(x: np.ndarray, lam: float) -> float:
    x = np.asarray(x, dtype=float)
    log_factorials = np.array([math.lgamma(k + 1) for k in x])
    return float(np.sum(x * np.log(lam) - lam - log_factorials))


# --- Bayesian estimation of a coin's bias (Beta-Bernoulli) ------------------------


def beta_posterior(heads: int, tails: int, a: float = 1.0, b: float = 1.0) -> tuple[float, float]:
    """Prior Beta(a, b) + coin flips -> posterior Beta(a + heads, b + tails).

    The Beta prior is "conjugate" to the Bernoulli likelihood: the posterior
    is in the same family, so updating is just adding counts.
    """
    return a + heads, b + tails


def beta_mean(a: float, b: float) -> float:
    """Posterior mean: the Bayesian point estimate (always strictly between 0 and 1)."""
    return a / (a + b)


def beta_map(a: float, b: float) -> float:
    """MAP = mode of the posterior, (a - 1) / (a + b - 2) for a, b > 1.

    With a uniform prior Beta(1, 1), MAP equals the MLE heads / (heads + tails).
    """
    return (a - 1) / (a + b - 2)
