"""Gaussian mixture model fitted with expectation-maximisation (full covariances)."""

from __future__ import annotations

import numpy as np

from s2_math.softmax_logsumexp.impl import logsumexp
from s3_classical_ml.kmeans.impl import KMeans


def gaussian_log_pdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """log N(x; mean, cov) for every row of X, via a Cholesky factor (no explicit inverse).

    With cov = L L^T:  (x - mu)^T cov^-1 (x - mu) = ||L^-1 (x - mu)||^2
    and log det(cov) = 2 * sum(log diag(L)).
    """
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (X - mean).T)  # (d, n)
    maha = (z**2).sum(axis=0)
    log_det = 2 * np.log(np.diag(L)).sum()
    return -0.5 * (d * np.log(2 * np.pi) + log_det + maha)


class GaussianMixture:
    """Model the data as coming from K Gaussian blobs:
        p(x) = sum_k pi_k N(x; mu_k, Sigma_k)

    EM alternates:
    E-step: responsibilities r_ik = P(component k | x_i) -- soft assignments.
    M-step: refit every component as a WEIGHTED Gaussian using those
            responsibilities as weights; pi_k = average responsibility.
    Each round never decreases the data log-likelihood.
    """

    def __init__(
        self,
        n_components: int = 1,
        max_iter: int = 200,
        tol: float = 1e-6,
        reg_covar: float = 1e-6,
        n_init: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.n_components, self.max_iter, self.tol = n_components, max_iter, tol
        self.reg_covar, self.n_init, self.random_state = reg_covar, n_init, random_state

    def _log_joint(self, X: np.ndarray) -> np.ndarray:
        """log pi_k + log N(x_i; mu_k, Sigma_k), shape (n, K)."""
        return np.column_stack(
            [
                np.log(self.weights_[k]) + gaussian_log_pdf(X, self.means_[k], self.covariances_[k])
                for k in range(self.n_components)
            ]
        )

    def _m_step(self, X: np.ndarray, R: np.ndarray) -> None:
        n, d = X.shape
        Nk = R.sum(axis=0) + 1e-300  # effective number of points per component
        self.weights_ = Nk / n
        self.means_ = (R.T @ X) / Nk[:, None]
        covs = []
        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = (R[:, k, None] * diff).T @ diff / Nk[k]
            covs.append(cov + self.reg_covar * np.eye(d))  # keep it invertible
        self.covariances_ = np.array(covs)

    def fit(self, X: np.ndarray) -> GaussianMixture:
        """Run EM `n_init` times from different k-means starts; keep the best likelihood."""
        X = np.asarray(X, dtype=float)
        seeds = np.random.default_rng(self.random_state).integers(0, 2**31, size=self.n_init)
        best = None
        for seed in seeds:
            self._fit_once(X, int(seed))
            if best is None or self.lower_bound_ > best["lower_bound_"]:
                best = {k: v for k, v in vars(self).items() if k.endswith("_")}
        vars(self).update(best)
        return self

    def _fit_once(self, X: np.ndarray, seed: int) -> None:
        # Start from a k-means partition (hard responsibilities), as sklearn does.
        labels = KMeans(self.n_components, n_init=1, random_state=seed).fit_predict(X)
        R = np.eye(self.n_components)[labels]
        self._m_step(X, R)
        self.log_likelihood_history_ = []
        self.converged_ = False
        self.n_iter_ = 0
        for _ in range(self.max_iter):
            self.n_iter_ += 1
            log_joint = self._log_joint(X)
            log_px = logsumexp(log_joint, axis=1)  # log p(x_i)
            self.log_likelihood_history_.append(float(log_px.mean()))
            R = np.exp(log_joint - log_px[:, None])  # E-step (in log space for stability)
            self._m_step(X, R)  # M-step
            hist = self.log_likelihood_history_
            if len(hist) > 1 and abs(hist[-1] - hist[-2]) < self.tol:
                self.converged_ = True
                break
        self.lower_bound_ = float(self.score(X))

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """log p(x) for each row."""
        return logsumexp(self._log_joint(np.asarray(X, dtype=float)), axis=1)

    def score(self, X: np.ndarray) -> float:
        """Mean log-likelihood per sample."""
        return float(self.score_samples(X).mean())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        log_joint = self._log_joint(np.asarray(X, dtype=float))
        return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def bic(self, X: np.ndarray) -> float:
        """Bayesian information criterion: lower is better. Penalises parameters."""
        n, d = np.asarray(X).shape
        K = self.n_components
        n_params = (K - 1) + K * d + K * d * (d + 1) / 2
        return float(-2 * self.score(X) * n + n_params * np.log(n))

    def sample(self, n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        comps = rng.choice(self.n_components, size=n, p=self.weights_)
        X = np.array([rng.multivariate_normal(self.means_[k], self.covariances_[k]) for k in comps])
        return X, comps
