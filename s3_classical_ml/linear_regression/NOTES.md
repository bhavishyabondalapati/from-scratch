# Linear regression (closed form + gradient descent)

## Intuition
Draw the straight line (or flat plane, with several features) that passes as close as possible to
the data points. "Close" means the sum of squared vertical gaps between the points and the line is
as small as possible. Each coefficient says "if this feature goes up by one unit, and the others
stay fixed, the prediction changes by this much".

## Core math
- Model: `ŷ = Xw + b`. Absorb `b` by adding a column of ones to `X`.
- Loss: `L(w) = (1/n) ‖Xw − y‖²`.
- Gradient: `∇L = (2/n) Xᵀ(Xw − y)`.
- Setting the gradient to zero gives the **normal equations** `XᵀX w = Xᵀy`, so
  `w = (XᵀX)⁻¹Xᵀy`.
- Geometric view: `Xw` is the projection of `y` onto the column space of `X`; the residual is
  orthogonal to every feature column.
- Gradient descent: `w ← w − η ∇L`. Converges iff `η < 2 / λ_max(2XᵀX/n)`.
- Probabilistic view: least squares is the MLE when `y = Xw + Gaussian noise`.
- `R² = 1 − SS_res / SS_tot`.

## Complexity
| Method | Time | Notes |
|---|---|---|
| Normal equations / QR / SVD | O(n d² + d³) | exact; needs all data in memory |
| Batch gradient descent | O(n d) per iteration | iterations grow with the condition number |
| SGD | O(d) per sample | scales to huge n and streaming data |

## Common pitfalls
- Inverting `XᵀX` when features are collinear (singular matrix): use `lstsq`/pseudo-inverse or ridge.
- Gradient descent on unscaled features: slow or divergent. Standardise first.
- Forgetting the intercept.
- Reading coefficients as importance when features have different scales or are correlated.
- Outliers dominate squared error; consider Huber loss.
- A high training R² says nothing about generalisation; evaluate on held-out data.
- Extrapolating far outside the range of the training data.

## When to use vs. alternatives
- First baseline for any regression task; interpretable; fast.
- Many/collinear features → ridge. Want feature selection → lasso.
- Nonlinear relationships → add polynomial/interaction features, or trees/boosting.
- Closed form when d is modest (up to ~10⁴); (S)GD when d or n is huge.

## Interview questions
1. **Derive the normal equations.** Differentiate `‖Xw − y‖²` to get `2Xᵀ(Xw − y) = 0`.
2. **When is the closed form a bad idea?** Very large d (O(d³)), data that does not fit in
   memory, or singular `XᵀX`.
3. **What are the assumptions of linear regression?** Linearity, independent errors, constant
   error variance (homoscedasticity), no perfect multicollinearity; normal errors for inference.
4. **Why squared error?** Smooth and convex with a closed form, and equals maximum likelihood
   under Gaussian noise. The downside is sensitivity to outliers.
5. **What happens with multicollinearity?** Coefficients become unstable (huge variance) though
   predictions may still be fine; regularise or drop features.
