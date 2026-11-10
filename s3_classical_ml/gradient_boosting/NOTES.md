# Gradient boosting (regression + classification)

## Intuition
Build a team of small trees **one after another**, where each new tree is trained on the
*mistakes of the team so far*.

1. Start with a constant guess (the mean).
2. Look at the errors (residuals): where is the current model too low or too high?
3. Fit a small tree that predicts those errors.
4. Add a *fraction* of that tree's output (the learning rate) to the model.
5. Repeat.

Each tree is weak (shallow), so it has high bias; adding hundreds of corrections drives the bias
down. This is the opposite strategy to a random forest, which averages many strong trees to
reduce variance.

## Core math
- Model: `F_M(x) = F₀ + η Σₘ hₘ(x)`.
- At round m compute **pseudo-residuals** `rᵢ = −∂L(yᵢ, F(xᵢ)) / ∂F(xᵢ)` and fit `hₘ` to them.
  This is gradient descent where the "parameter" is the prediction for each training point.
- Squared loss `½(y − F)²`: `r = y − F` (ordinary residuals); `F₀ = mean(y)`.
- Log-loss with `p = σ(F)`: `r = y − p`; `F₀ = log(p̄ / (1 − p̄))`; leaf values use a Newton step
  `γ = Σ r / Σ p(1 − p)` over the samples in the leaf.
- Regularisation: learning rate `η` (shrinkage), number of trees, tree depth, row subsampling
  (stochastic GB), plus L1/L2 on leaf values in XGBoost-style implementations.

## Complexity
Train: O(M · d · n log n · depth), sequential across trees (cannot be parallelised over trees).
Predict: O(M · depth). Histogram-based versions (LightGBM, XGBoost `hist`) bin features so a
split search costs O(#bins) instead of O(n).

## Common pitfalls
- **Overfitting with too many trees** (unlike random forests): use early stopping on a
  validation set.
- Learning rate and number of trees trade off: smaller η needs more trees but generalises better.
- Deep trees in boosting overfit quickly; depth 3–8 is typical.
- Sensitive to noisy labels/outliers with squared loss (each round chases the big residuals);
  use Huber or absolute loss.
- For classification, forgetting the leaf-value Newton step gives a slow, poorly scaled model.
- Like all trees: no extrapolation.

## When to use vs. alternatives
- State of the art for tabular data (XGBoost / LightGBM / CatBoost), ranking, and many Kaggle
  problems.
- Random forest: easier to tune, parallel, harder to overfit; a bit less accurate.
- Linear models: tiny data, need for extrapolation or coefficients.
- Deep learning: unstructured data (images, text, audio).

## Interview questions
1. **Why is it called *gradient* boosting?** Each tree is fitted to the negative gradient of the
   loss with respect to the current predictions: functional gradient descent.
2. **Bagging vs boosting in terms of bias/variance?** Bagging lowers variance of low-bias
   learners; boosting lowers bias of high-bias (weak) learners.
3. **What does the learning rate do?** Shrinks each tree's contribution so later trees can
   correct it; lower rate + more trees usually generalises better.
4. **How does XGBoost differ from classic GBM?** Second-order (Newton) objective, explicit
   regularisation of leaf weights and tree size, histogram/sparse-aware split finding, parallelism.
5. **How do you prevent overfitting?** Early stopping, shrinkage, shallow trees, subsampling
   rows/columns, min samples per leaf.
