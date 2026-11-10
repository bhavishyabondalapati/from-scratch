# PCA as a model

The derivation (eigendecomposition vs SVD) is in [`s2_math/pca`](../../s2_math/pca/NOTES.md).
This folder wraps it as a reusable `fit / transform / inverse_transform` model.

## Intuition
Rotate the data so the first axis follows the biggest spread, the second the next biggest,
and so on; then drop the axes with little spread. Used as a **preprocessing step**: fewer,
uncorrelated features that keep most of the information.

## Core math
- Fit on training data only: mean `μ`, components `Vₖ` (rows), variances `λ`.
- `transform(x) = (x − μ) Vₖᵀ`; with **whitening** divide each coordinate by `√λᵢ`, so every
  output feature has unit variance and they are uncorrelated (covariance = identity).
- `inverse_transform(z) = z Vₖ + μ` (undo whitening first).
- Choosing k by a variance target τ: smallest k with `Σᵢ≤ₖ λᵢ / Σλ ≥ τ`.

## Complexity
Fit O(min(n d², n² d)); transform O(n d k).

## Common pitfalls
- Fitting PCA on all data before the train/test split (leakage).
- Whitening amplifies noise in low-variance directions.
- Interpreting components as "real" factors; they are just orthogonal variance directions.
- Forgetting to standardise features with different units (do it before PCA).
- Signs of components flip between libraries/runs.

## When to use vs. alternatives
- Speed up / regularise models on many correlated features; compress images; denoise; 2D plots.
- Visualising non-linear structure → t-SNE / UMAP. Sparse non-negative data → NMF / truncated SVD.

## Interview questions
1. **Why fit PCA only on the training set?** Test statistics would leak into the projection.
2. **What does whitening do?** Makes the transformed features uncorrelated with unit variance.
3. **How many components?** Explained-variance threshold (e.g. 95%), scree elbow, or CV on the
   downstream task.
4. **Does PCA help every model?** No; it can discard low-variance directions that are highly
   predictive of the label, and tree models do not need decorrelated inputs.
5. **What is the reconstruction error?** `(n − 1) Σ_{i>k} λᵢ`: the variance in dropped directions.
