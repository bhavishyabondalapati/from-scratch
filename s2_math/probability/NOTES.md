# Basic probability: MLE and Bayes

## Intuition
**Bayes' rule** is how to update a belief when evidence arrives:
*new belief ∝ old belief × how well the hypothesis explains the evidence*. The famous surprise:
a 99%-accurate test for a rare disease still produces mostly false alarms, because healthy people
vastly outnumber sick ones.

**Maximum likelihood estimation (MLE)** answers "which parameter value makes the data I saw most
probable?" Flip a coin 10 times and see 7 heads: the bias that best explains that is 0.7.
Training almost any ML model is MLE in disguise: minimising cross-entropy = maximising the
likelihood of the labels; minimising squared error = MLE under Gaussian noise.

**MAP** adds a prior belief about the parameter; it is MLE plus a regulariser.

## Core math
- Bayes: `P(H|E) = P(E|H) P(H) / P(E)`, with `P(E) = Σ_h P(E|h) P(h)`.
  posterior ∝ likelihood × prior.
- Likelihood of i.i.d. data: `L(θ) = Π p(xᵢ | θ)`; work with `log L = Σ log p(xᵢ | θ)`
  (sums are stable and easy to differentiate).
- MLE: `θ̂ = argmax log L(θ)`; solve `∂ log L / ∂θ = 0`.

| Model | MLE |
|---|---|
| Bernoulli(p) | `p̂ = mean(x)` |
| Gaussian(μ, σ²) | `μ̂ = mean(x)`, `σ̂² = mean((x − μ̂)²)` (divisor n: biased) |
| Poisson(λ) | `λ̂ = mean(x)` |
| Exponential(rate) | `1 / mean(x)` |
| Categorical | normalised counts |

- MAP: `θ̂ = argmax [log L(θ) + log p(θ)]`. Gaussian prior ⇒ L2 penalty; Laplace prior ⇒ L1.
- Conjugacy: Beta(a, b) prior + h heads, t tails ⇒ Beta(a + h, b + t) posterior.
  Mean `a/(a+b)`, mode `(a−1)/(a+b−2)`.

## Complexity
Closed-form MLEs are O(n). When no closed form exists (logistic regression, neural nets),
maximise the log-likelihood with gradient descent.

## Common pitfalls
- **Base-rate neglect:** confusing `P(positive | sick)` with `P(sick | positive)`.
- Multiplying many probabilities underflows to 0: add logs instead.
- MLE with little data overfits: zero counts give probability 0 (then `log 0 = −∞`). Use
  smoothing / a prior.
- MLE variance uses divisor n (biased); the unbiased estimator uses n − 1.
- Assuming independence (i.i.d.) when samples are correlated.
- Likelihood is a function of θ, not a probability distribution over θ.

## When to use vs. alternatives
- MLE: plenty of data, want a simple point estimate (the default in ML).
- MAP: want regularisation / have prior knowledge; still a point estimate.
- Full Bayesian posterior: need uncertainty (small data, decision-making, A/B tests, Thompson
  sampling in `s6_rl/bandits`).

## Interview questions
1. **A test is 99% sensitive with a 5% false-positive rate, prevalence is 1%. P(sick | positive)?**
   `0.99·0.01 / (0.99·0.01 + 0.05·0.99) ≈ 16.7%`.
2. **Why maximise the log-likelihood instead of the likelihood?** Same argmax (log is
   monotonic), products become sums, no underflow, easier derivatives.
3. **Derive the MLE for a Bernoulli.** `log L = h log p + t log(1−p)`; derivative
   `h/p − t/(1−p) = 0` ⇒ `p = h/(h+t)`.
4. **MLE vs MAP?** MAP multiplies the likelihood by a prior; with a flat prior they coincide, and
   with lots of data the prior's influence vanishes.
5. **How is cross-entropy loss related to MLE?** Minimising cross-entropy is maximising the
   log-likelihood of the labels under the model's predicted distribution.
