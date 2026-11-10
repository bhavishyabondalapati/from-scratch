# Support vector machines (subgradient + SMO, kernel trick)

## Intuition
Many lines can separate two classes. The SVM picks the one with the **widest empty street**
between the classes (the *margin*). A wide street means small wobbles in the data do not flip
predictions, so it generalises well. Only the points touching the edge of the street matter;
they are the **support vectors**. Delete any other point and the answer does not change.

Real data is messy, so the **soft margin** lets some points sit inside the street or on the wrong
side, for a penalty controlled by `C` (large C = few violations tolerated, narrow street).

The **kernel trick**: if no straight line works, map the data to a richer space where one does.
The dual problem only ever uses dot products `xᵢ·xⱼ`, so replacing them with a kernel function
`K(xᵢ, xⱼ)` works *as if* we had mapped to a huge (even infinite) feature space, without ever
computing the mapping.

## Core math
- Primal: `min ½‖w‖² + C Σ ξᵢ` s.t. `yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ`, `ξᵢ ≥ 0`.
  Equivalently hinge loss: `min (λ/2)‖w‖² + (1/n) Σ max(0, 1 − yᵢ f(xᵢ))` with `C = 1/(λn)`.
- Margin width = `2 / ‖w‖`.
- Subgradient of hinge: `−yᵢxᵢ` if `yᵢ f(xᵢ) < 1`, else 0.
- Dual: `max Σαᵢ − ½ ΣΣ αᵢαⱼyᵢyⱼK(xᵢ, xⱼ)` s.t. `0 ≤ αᵢ ≤ C`, `Σαᵢyᵢ = 0`.
- Decision function: `f(x) = Σᵢ αᵢyᵢK(xᵢ, x) + b`.
- KKT conditions: `αᵢ = 0 ⇒ yᵢfᵢ ≥ 1`; `0 < αᵢ < C ⇒ yᵢfᵢ = 1`; `αᵢ = C ⇒ yᵢfᵢ ≤ 1`.
- SMO: update two multipliers at a time (the smallest move that keeps `Σαᵢyᵢ = 0`); the
  two-variable problem is a clipped 1D quadratic.
- Kernels: linear `a·b`; polynomial `(γ a·b + c)ᵈ`; RBF `exp(−γ‖a − b‖²)`.

## Complexity
| | Train | Predict |
|---|---|---|
| Linear (subgradient / coordinate descent) | O(n·d) per pass | O(d) |
| Kernel SMO | between O(n²) and O(n³); kernel matrix is O(n²) memory | O(#SV · d) |

## Common pitfalls
- **Feature scaling is essential** (distances and dot products).
- RBF `γ` too large → every point is its own island (overfit); too small → nearly linear.
  Tune `C` and `γ` together on a log grid.
- Kernel SVMs do not scale past ~10⁵ samples.
- Outputs are scores, not probabilities (Platt scaling adds them afterwards).
- Multi-class needs one-vs-rest or one-vs-one wrappers.
- Subgradient descent is not monotone: keep the best iterate or average.

## When to use vs. alternatives
- Small/medium datasets, high-dimensional features (text with a linear kernel), clear margin.
- Large tabular data → gradient boosting. Need probabilities → logistic regression.
- Unstructured data at scale → neural networks (which largely displaced kernel SVMs).

## Interview questions
1. **What are support vectors?** Training points with `α > 0`: on or inside the margin; they
   alone define the boundary.
2. **What does C control?** The trade-off between a wide margin and few margin violations
   (inverse of regularisation strength).
3. **What is the kernel trick and why does it work?** The dual depends on the data only through
   inner products, so any positive semi-definite kernel can stand in for an inner product in
   an implicit feature space.
4. **Hinge loss vs log-loss?** Hinge is exactly zero beyond the margin (sparse solutions, no
   probabilities); log-loss never reaches zero and yields calibrated probabilities.
5. **Why does SMO optimise two variables at a time?** The constraint `Σαᵢyᵢ = 0` means one
   variable cannot move alone; two is the smallest feasible step and has a closed-form solution.
