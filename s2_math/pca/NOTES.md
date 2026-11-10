# PCA via SVD and via eigendecomposition

## Intuition
You have data with many features, and many of them move together. PCA finds new axes for the
data: the first axis points along the direction where the data is **most spread out**, the second
is the most spread-out direction perpendicular to the first, and so on. Keep only the first few
axes and you have a lower-dimensional version of the data that loses as little information
(variance) as possible.

Picture a cloud of points shaped like a flattened rugby ball: PCA finds the long axis, then the
medium axis, then the short one. Dropping the short one barely changes the cloud.

## Core math
1. Centre: `Xc = X − mean`.
2. Covariance: `C = XcᵀXc / (n − 1)` (d × d).
3. **Eigen route:** `C v = λ v`. Eigenvectors `v` = principal directions, eigenvalues `λ` =
   variance along them.
4. **SVD route:** `Xc = U S Vᵀ`. Then `C = V (S²/(n−1)) Vᵀ`, so the right singular vectors are the
   same directions and `λᵢ = sᵢ² / (n − 1)`.
5. Project: `Z = Xc Vₖ` (n × k). Reconstruct: `X̂ = Z Vₖᵀ + mean`.
6. Explained variance ratio of component i: `λᵢ / Σ λ`.
7. Optimality (Eckart-Young): among all rank-k linear projections, PCA minimises the squared
   reconstruction error, which equals `(n − 1) · Σ_{i>k} λᵢ`.

## Complexity
| Route | Time | Notes |
|---|---|---|
| Covariance + eigh | O(n d² + d³) | fine when d is small |
| SVD of Xc | O(min(n d², n² d)) | more accurate; no `XᵀX` |
| Randomized / truncated SVD | ~O(n d k) | large data, small k |

## Common pitfalls
- **Not centring** the data: the first component just points at the mean.
- **Not scaling** features with different units: the feature with the largest numbers dominates.
  Standardise first unless the units are comparable.
- Signs of components are arbitrary (`v` and `−v` are both valid); compare up to sign.
- Fitting PCA on train + test together leaks information; fit on train, apply to test.
- PCA is linear and variance-based: high variance ≠ useful for the label, and curved structure
  (a spiral) is not captured.
- Forming `XᵀX` squares the condition number; tiny eigenvalues come out inaccurate or negative.

## When to use vs. alternatives
- Compression, denoising, decorrelating features, speeding up downstream models, 2D plots.
- Visualising clusters / nonlinear structure → t-SNE, UMAP.
- Supervised dimensionality reduction → LDA. Non-negative parts → NMF. Nonlinear → kernel PCA,
  autoencoders (a linear autoencoder with squared loss learns the PCA subspace).

## Interview questions
1. **What do the eigenvalues and eigenvectors of the covariance matrix represent?** Directions
   of maximal variance and the variance along each.
2. **Why is PCA usually computed with SVD?** It avoids forming `XᵀX` (better numerical accuracy)
   and gives the same components.
3. **How do you choose k?** Cumulative explained variance (e.g. 95%), a scree-plot elbow, or
   cross-validating the downstream task.
4. **Do you need to standardise before PCA?** Yes when features have different scales/units;
   otherwise the result reflects units rather than structure.
5. **How does PCA relate to an autoencoder?** A linear autoencoder trained with MSE spans the
   same subspace as the top-k principal components.
