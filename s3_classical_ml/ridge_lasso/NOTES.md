# Ridge and lasso regression

## Intuition
Plain linear regression will happily use huge positive and negative weights to chase noise,
especially with many or correlated features. **Regularisation** adds a cost for large weights, so
the model only uses a big weight when the data strongly demands it.

- **Ridge (L2):** penalise the *squared* weights. Shrinks every weight toward zero a bit, spreads
  weight across correlated features. Nothing becomes exactly zero.
- **Lasso (L1):** penalise the *absolute* weights. Pushes many weights to **exactly zero**, so it
  also selects features.

Why the difference? The L2 penalty's pull (`2αw`) fades as a weight gets small, so weights never
quite reach zero. The L1 penalty pulls with constant force `α` however small the weight is, so
weak features get pushed all the way to zero. Geometrically, the L1 ball is a diamond with
corners on the axes, and the loss contours tend to touch it at a corner.

## Core math
- Ridge: `min ‖y − Xw‖² + α‖w‖²` ⇒ `w = (XᵀX + αI)⁻¹Xᵀy`.
  In SVD terms each direction is shrunk by `s² / (s² + α)`: weak directions are damped most.
- Lasso: `min (1/2n)‖y − Xw‖² + α‖w‖₁`. Coordinate update:
  `wⱼ = S(ρⱼ, α) / zⱼ`, with `ρⱼ = (1/n) xⱼᵀ(y − X₋ⱼw₋ⱼ)`, `zⱼ = (1/n)‖xⱼ‖²`,
  and soft threshold `S(ρ, α) = sign(ρ) · max(|ρ| − α, 0)`.
- All weights are zero when `α ≥ α_max = max|Xᵀy| / n`.
- Bayesian view: ridge = MAP with a Gaussian prior on `w`; lasso = MAP with a Laplace prior.
- Elastic net = both penalties together.

## Complexity
| Model | Time |
|---|---|
| Ridge closed form | O(n d² + d³) |
| Lasso coordinate descent | O(n d) per sweep; usually tens of sweeps |

## Common pitfalls
- Not standardising features: the penalty treats all weights alike, so features on large
  scales are penalised less.
- Penalising the intercept (shifts predictions for no reason).
- Different libraries scale the objective differently (`1/2n` or not), so "α = 1" is not portable.
- Lasso with correlated features picks one almost arbitrarily; selection is unstable.
- Treating lasso's selected features as causal/important without checking stability.
- Choosing α on the test set; use cross-validation.

## When to use vs. alternatives
- Ridge: many correlated features, d > n, you want stable predictions.
- Lasso: you believe only a few features matter, or you want a compact, interpretable model.
- Elastic net: correlated groups *and* sparsity.
- Nonlinear interactions → tree ensembles.

## Interview questions
1. **Why does L1 give sparsity and L2 not?** L1's gradient has constant magnitude near zero (and
   the constraint region has corners on the axes); L2's gradient vanishes near zero.
2. **How does ridge fix multicollinearity?** `XᵀX + αI` is always invertible and well
   conditioned; variance of the estimates drops at the cost of some bias.
3. **Why can't lasso be solved in closed form or with plain GD?** `|w|` is not differentiable at
   zero; use coordinate descent, proximal gradient (ISTA), or LARS.
4. **What does regularisation do to bias and variance?** Increases bias, decreases variance; the
   best α balances them (found by cross-validation).
5. **What is the Bayesian interpretation?** Penalised least squares is MAP estimation: Gaussian
   prior ⇒ ridge, Laplace prior ⇒ lasso.
