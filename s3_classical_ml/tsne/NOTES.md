# t-SNE (small-scale, exact)

## Intuition
t-SNE draws a 2D map of high-dimensional data where **points that were neighbours stay
neighbours**. It does not try to preserve large distances, only "who is close to whom".

1. In the original space, turn distances into probabilities: "if point i picked a neighbour at
   random, favouring close ones, how likely is j?"
2. In the 2D map, define the same kind of probabilities.
3. Move the 2D points until the two sets of probabilities match.

The 2D similarities use a heavy-tailed Student-t curve instead of a Gaussian. In 2D there is
not enough room for every moderately-close point to stay moderately close (the "crowding
problem"); heavy tails let such points sit farther apart, which separates clusters cleanly.

## Core math
- `p_{j|i} = exp(−‖xᵢ − xⱼ‖²/2σᵢ²) / Σₖ exp(−‖xᵢ − xₖ‖²/2σᵢ²)`, σᵢ found by binary search so the
  perplexity `2^{H(Pᵢ)}` equals the user's value (≈ effective number of neighbours).
- Symmetrise: `pᵢⱼ = (p_{j|i} + p_{i|j}) / 2n`.
- `qᵢⱼ = (1 + ‖yᵢ − yⱼ‖²)⁻¹ / Σ_{k≠l} (1 + ‖y_k − y_l‖²)⁻¹`.
- Cost `C = KL(P‖Q) = Σ pᵢⱼ log(pᵢⱼ/qᵢⱼ)`;
  gradient `∂C/∂yᵢ = 4 Σⱼ (pᵢⱼ − qᵢⱼ)(yᵢ − yⱼ)(1 + ‖yᵢ − yⱼ‖²)⁻¹`.
- Tricks: early exaggeration (multiply P by ~12 at first), momentum, adaptive gains.

## Complexity
Exact: O(n²) time and memory per iteration (fine up to a few thousand points). Barnes-Hut:
O(n log n). FFT-accelerated (openTSNE/FIt-SNE): ~O(n).

## Common pitfalls
- **Reading too much into the map:** cluster sizes and between-cluster distances are not
  meaningful; only local neighbourhoods are.
- Perplexity changes the picture a lot; try several (5–50).
- Different runs/seeds give different layouts.
- Cannot embed new points (no `transform`); it is for visualisation, not features.
- Run PCA to ~50 dimensions first on very high-dimensional data.

## When to use vs. alternatives
- Visualising clusters in embeddings, images, single-cell data.
- UMAP: faster, preserves more global structure, can transform new points.
- PCA: linear, deterministic, meaningful distances, but misses non-linear structure.

## Interview questions
1. **Why Student-t in the low-dimensional space?** Heavy tails solve the crowding problem by
   allowing moderate distances to become large ones.
2. **What is perplexity?** A smooth measure of the effective number of neighbours per point;
   it sets each σᵢ.
3. **Can you trust distances between clusters in a t-SNE plot?** No; only local structure is
   preserved.
4. **Why is KL(P‖Q) asymmetric and why does that matter?** It heavily penalises putting
   neighbours far apart (large p, small q) but barely penalises putting non-neighbours close.
5. **t-SNE vs PCA vs UMAP?** PCA is linear/global; t-SNE is non-linear/local and slow; UMAP is
   non-linear, faster, and keeps more global structure.
