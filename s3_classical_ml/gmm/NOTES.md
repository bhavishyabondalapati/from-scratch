# Gaussian mixture model with EM

## Intuition
Imagine the data was made by first secretly rolling a weighted die to pick one of K "blob
makers", then drawing a point from that maker's bell-shaped (Gaussian) cloud. A GMM tries to
recover the makers: their centres, shapes (covariances), and how often each was picked.

Unlike k-means, every point is assigned **softly**: "70% blob A, 30% blob B". And blobs may be
stretched ellipses of different sizes, not just equal circles.

**EM** (expectation-maximisation) is the chicken-and-egg solver:
- **E-step:** given the current blobs, how responsible is each blob for each point?
- **M-step:** given those responsibilities, refit each blob as a weighted average.

## Core math
- `p(x) = Σₖ πₖ N(x; μₖ, Σₖ)`.
- E-step: `rᵢₖ = πₖ N(xᵢ; μₖ, Σₖ) / Σⱼ πⱼ N(xᵢ; μⱼ, Σⱼ)` (compute in log space with log-sum-exp).
- M-step with `Nₖ = Σᵢ rᵢₖ`: `πₖ = Nₖ/n`, `μₖ = Σᵢ rᵢₖ xᵢ / Nₖ`,
  `Σₖ = Σᵢ rᵢₖ (xᵢ − μₖ)(xᵢ − μₖ)ᵀ / Nₖ`.
- EM maximises a lower bound on the log-likelihood (the ELBO); each iteration never decreases
  `log p(X)`. It converges to a local optimum.
- Log-density via Cholesky `Σ = LLᵀ`: Mahalanobis distance `‖L⁻¹(x − μ)‖²`, `log|Σ| = 2Σ log Lᵢᵢ`.
- Model selection: BIC `= −2 log L + p log n`.

## Complexity
O(n · K · d²) per iteration for the E-step (plus O(K · d³) for the factorisations); memory O(n·K).

## Common pitfalls
- **Singular covariances:** a component collapses onto one point and the likelihood goes to
  infinity. Add a small value to the diagonal (`reg_covar`).
- Local optima: initialise from k-means and/or use several restarts.
- Underflow when multiplying tiny densities: stay in log space.
- Too many parameters in high dimensions with full covariances; use diagonal/tied covariances.
- K must be chosen (BIC/AIC, held-out likelihood).

## When to use vs. alternatives
- Soft clustering, elliptical clusters of different sizes, density estimation, anomaly detection
  (low `p(x)`), generative sampling.
- Hard, spherical clusters and speed → k-means. Arbitrary shapes → DBSCAN / spectral.
- Complex high-dimensional densities → normalising flows, VAEs.

## Interview questions
1. **What do the E and M steps do?** E computes posterior responsibilities; M re-estimates
   weights, means, covariances by responsibility-weighted MLE.
2. **Why does EM increase the likelihood?** It maximises a lower bound that touches the true
   log-likelihood at the current parameters, so improving the bound improves the likelihood.
3. **GMM vs k-means?** GMM: soft assignments, full covariances, probabilistic; k-means is its
   hard-assignment, spherical, equal-variance limit.
4. **What can go wrong with EM?** Local optima, collapsing components (singular covariance),
   slow convergence.
5. **How do you choose the number of components?** BIC/AIC or held-out log-likelihood.
