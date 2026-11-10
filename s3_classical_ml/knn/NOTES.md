# k-nearest neighbours

## Intuition
"Tell me who your neighbours are and I will tell you who you are." To label a new point, find
the `k` training points closest to it and let them vote (classification) or average their values
(regression). There is no training step at all: the model *is* the data.

- Small `k`: wiggly boundary, follows noise (low bias, high variance). `k = 1` memorises.
- Large `k`: smooth boundary, may blur real structure (high bias, low variance). `k = n`
  always predicts the majority class.

## Core math
- Distance, usually Euclidean: `d(a, b) = √Σ(aᵢ − bᵢ)²`. Minkowski: `(Σ|aᵢ − bᵢ|ᵖ)^{1/p}`.
- Vectorised trick: `‖a − b‖² = ‖a‖² − 2a·b + ‖b‖²`.
- Classification: `ŷ = argmax_c Σ_{i ∈ N_k(x)} wᵢ · 1[yᵢ = c]`; regression: weighted mean.
- Weights: uniform (`wᵢ = 1`) or inverse distance (`wᵢ = 1/dᵢ`).
- Asymptotically, 1-NN's error is at most twice the Bayes-optimal error.

## Complexity
| Step | Brute force | KD-tree / ball tree |
|---|---|---|
| Fit | O(1) (store) | O(n log n) |
| Query | O(n·d) per point | ~O(log n) in low d, degrades to O(n) in high d |
| Memory | O(n·d) | O(n·d) |

## Common pitfalls
- **Feature scaling:** a feature measured in thousands dominates one measured in fractions.
  Always standardise.
- **Curse of dimensionality:** in high dimensions all points are almost equally far apart, so
  "nearest" loses meaning. Reduce dimension first.
- Irrelevant features add noise to every distance.
- Slow prediction and large memory on big datasets (the opposite of most models).
- Ties in votes (use odd `k` for binary problems) and ties in distance.
- Imbalanced classes: the majority class crowds the vote; use distance weighting.

## When to use vs. alternatives
- Small/medium, low-dimensional data with irregular decision boundaries; quick baseline;
  recommender-style "similar items".
- Large n or high d → approximate nearest neighbours (HNSW, IVF; see `s5_nlp/dense_retrieval`).
- Need fast inference or interpretability → trees, linear models.

## Interview questions
1. **How do you choose k?** Cross-validation; small k overfits, large k underfits; odd k avoids
   binary ties. A common starting point is √n.
2. **Why must features be scaled?** Distances add up raw differences, so large-scale features
   dominate.
3. **What is the curse of dimensionality for KNN?** Distances concentrate; the volume to cover
   grows exponentially, so neighbours are no longer "near".
4. **Training and prediction complexity?** Training O(1); brute-force prediction O(n·d) per
   query: the cost is paid at inference.
5. **Is KNN parametric?** No. It is non-parametric and instance-based: model capacity grows
   with the data.
