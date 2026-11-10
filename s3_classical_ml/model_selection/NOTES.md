# Model selection: splits, k-fold CV, grid search

## Intuition
A model's score on the data it trained on is optimistic, like grading a student on questions they
saw in advance. To estimate real-world performance we must test on data the model never saw.

- **Train / validation / test:** fit on train, compare settings on validation, report test
  **once** at the end. If you look at test to make decisions, it is no longer a fair exam.
- **k-fold cross-validation:** with little data, holding out one fixed chunk is wasteful and
  noisy. Split into k folds; train on k − 1 and validate on the remaining one, k times; average.
- **Stratified k-fold:** keep class proportions the same in every fold (vital for imbalance).
- **Grid search:** try every combination of hyperparameters, score each with CV, keep the best.

## Core math
- CV estimate: `CV = (1/k) Σⱼ score(model trained without fold j, fold j)`.
- Bias/variance of the estimate: larger k → less pessimistic (more training data per fit) but
  more compute and correlated folds; k = 5 or 10 is standard. Leave-one-out is k = n.
- Grid size = product of the number of values per hyperparameter (grows exponentially).

## Complexity
Grid search: `(#combinations) × k × cost(fit)`.

## Common pitfalls
- **Data leakage:** scaling/PCA/feature selection fitted on the full data before splitting.
  Fit preprocessing inside each fold.
- Reporting the best CV score as the final performance (it is optimistically biased by the
  selection itself; use a held-out test set or nested CV).
- Shuffling time series (use forward-chaining splits) or splitting correlated samples (the same
  patient in train and test → group k-fold).
- Different folds for different candidates → scores not comparable.
- Grid search over many dimensions: random search is usually more efficient.

## When to use vs. alternatives
- Large data: a single train/val/test split is enough.
- Small data: k-fold (stratified for classification); nested CV for an unbiased estimate.
- Many hyperparameters/expensive models: random search, Bayesian optimisation, Hyperband.

## Interview questions
1. **Why separate validation and test sets?** Validation is used to make choices and becomes
   biased; the test set gives an untouched final estimate.
2. **Why stratify?** To keep class ratios stable across folds, reducing variance in the estimate,
   especially for rare classes.
3. **What is data leakage? Give an example.** Information from validation/test reaching training,
   e.g. standardising with the mean of the whole dataset.
4. **Grid vs random search?** Random search covers important dimensions better for the same
   budget when only a few hyperparameters matter.
5. **What is nested CV?** An outer CV for estimating performance and an inner CV for tuning, so
   the tuning does not bias the estimate.
