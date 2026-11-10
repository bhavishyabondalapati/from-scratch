# Naive Bayes (Gaussian + multinomial)

## Intuition
A **generative** classifier: instead of learning a boundary directly, learn what each class
"looks like", then ask which class would most plausibly have produced the new example.

Bayes' rule turns "how likely is this email given spam?" into "how likely is spam given this
email?". The **naive** part is assuming the features are independent once you know the class
(the word "free" tells you nothing about the word "money" beyond what the label already does).
That is rarely true, yet the classifier works well, because classification only needs the right
class to come out on top, not accurate probabilities.

- **Gaussian NB:** continuous features; a bell curve per (class, feature).
- **Multinomial NB:** counts (word frequencies); a word distribution per class.

## Core math
- `P(c | x) ∝ P(c) · Πⱼ P(xⱼ | c)`; work in logs: `log P(c) + Σⱼ log P(xⱼ | c)`.
- Gaussian: `log P(xⱼ | c) = −½ log(2πσ²_{cj}) − (xⱼ − μ_{cj})² / (2σ²_{cj})`.
- Multinomial: `P(word j | c) = (N_{cj} + α) / (N_c + α·V)` with Laplace smoothing `α`;
  score `= log P(c) + Σⱼ xⱼ log P(j | c)`, which is **linear in the counts**.
- Normalise with log-sum-exp to get posteriors.
- Fitting is MLE (or MAP with smoothing) by counting: no iterative optimisation.

## Complexity
Fit: O(n·d), a single pass. Predict: O(d·K) per example. Memory: O(d·K).

## Common pitfalls
- **Zero counts** without smoothing: one unseen word makes the whole class probability zero.
- Multiplying probabilities instead of summing logs: underflow on long documents.
- **Probabilities are poorly calibrated** (pushed toward 0 and 1) because correlated features
  are counted as independent evidence; trust the ranking, not the number.
- Gaussian NB on features that are far from normal (heavy skew, multi-modal).
- Multinomial NB requires non-negative features; TF-IDF works in practice, standardised
  features do not.

## When to use vs. alternatives
- Text classification baselines (spam, sentiment, topic), very high-dimensional sparse data,
  tiny training sets, streaming/online updates (just update the counts).
- Enough data and correlated features → logistic regression usually wins (discriminative).
- Need calibrated probabilities → logistic regression or calibrate NB afterwards.

## Interview questions
1. **What is the "naive" assumption?** Features are conditionally independent given the class.
2. **Why does it work despite the assumption being false?** The argmax can be right even when
   the probabilities are wrong; errors in dependence often affect all classes similarly.
3. **Why Laplace smoothing?** To avoid zero probabilities for unseen feature/class pairs; it is
   a MAP estimate under a Dirichlet prior.
4. **Generative vs discriminative?** NB models `P(x, y)` and converges fast with little data;
   logistic regression models `P(y | x)` and has lower asymptotic error with more data.
5. **Why is multinomial NB a linear classifier?** Its log-posterior is `b_c + w_c·x` with
   `w_c = log P(feature | c)`.
