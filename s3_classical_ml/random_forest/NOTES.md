# Random forest

## Intuition
One deep decision tree is a nervous expert: it fits its training data perfectly and changes its
mind completely if the data changes a little (high variance). A random forest asks **hundreds of
such experts, each trained a bit differently, and averages their answers**. Individual mistakes
point in different directions and cancel; what they agree on survives.

Two sources of deliberate randomness make the trees different:
1. **Bagging** (bootstrap aggregating): each tree trains on a random sample of rows drawn with
   replacement.
2. **Random feature subsets:** at every split a tree may only choose among a random handful of
   features (typically √d). This stops every tree from using the same strong feature at the top,
   which would make them all alike.

## Core math
- Averaging B identically distributed predictors with variance σ² and pairwise correlation ρ:
  `Var(average) = ρσ² + (1 − ρ)σ²/B`. More trees shrink the second term; **decorrelating**
  trees (feature subsampling) shrinks the first. Bias stays about the same as one tree.
- A bootstrap sample of size n contains each row with probability `1 − (1 − 1/n)ⁿ → 1 − 1/e ≈ 63.2%`.
- **Out-of-bag (OOB) estimate:** predict each row using only trees that did not train on it;
  behaves like built-in cross-validation.
- Classification: average class probabilities (soft vote). Regression: average predictions.

## Complexity
Train: O(B · m · n log n · depth) with m features tried per split; trees are independent, so
training parallelises perfectly. Predict: O(B · depth). Memory: B full trees.

## Common pitfalls
- Expecting more trees to overfit: they do not; error plateaus. (Deeper trees can.)
- Setting `max_features = d`: that is plain bagging; trees become correlated.
- Impurity-based feature importances favour high-cardinality/continuous features and split
  credit among correlated ones; prefer permutation importance.
- Poor extrapolation (cannot predict beyond the training target range).
- Large models and slow inference versus a single linear model.
- Imbalanced data: use class weights or balanced bootstraps.

## When to use vs. alternatives
- Excellent default for tabular data: little tuning, robust to outliers and unscaled or
  irrelevant features, handles non-linear interactions.
- Gradient boosting usually edges it out in accuracy when carefully tuned, but is more sensitive
  to hyperparameters and trains sequentially.
- Very high-dimensional sparse data (text) → linear models. Images/audio/text → deep learning.

## Interview questions
1. **Bagging vs boosting?** Bagging trains strong learners independently and averages them to
   cut variance; boosting trains weak learners sequentially to cut bias.
2. **Why sample features at each split?** To decorrelate the trees; otherwise a dominant feature
   makes all trees similar and averaging helps little.
3. **What is the OOB error?** Validation error computed on rows left out of each tree's
   bootstrap sample (~37%), with no extra data.
4. **Does a random forest overfit as you add trees?** No; adding trees only reduces variance.
5. **Which hyperparameters matter most?** `max_features`, tree size (`max_depth` /
   `min_samples_leaf`), and enough trees for the error to plateau.
