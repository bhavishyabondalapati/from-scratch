# Logistic regression (binary + multinomial)

## Intuition
Despite the name, it is a **classifier**. Take a linear score `w·x + b` (any real number) and
squash it through an S-shaped curve (the sigmoid) so it becomes a probability between 0 and 1.
A score of 0 means 50/50; large positive means "almost surely class 1".

The linear score is the **log-odds**: each feature adds or subtracts evidence, and a weight of
0.7 means "one more unit of this feature multiplies the odds by e^0.7 ≈ 2".

With more than two classes, compute one score per class and use **softmax** instead of sigmoid.
It is exactly a neural network with no hidden layer.

## Core math
- Binary: `p = σ(w·x + b)`, `σ(z) = 1/(1 + e^{−z})`.
- Loss (negative log-likelihood / log-loss / cross-entropy):
  `L = −(1/n) Σ [y log p + (1−y) log(1−p)]`.
- Gradient: `∇_w L = (1/n) Xᵀ(p − y)`, `∂L/∂b = mean(p − y)`. Same form as linear regression.
- The loss is **convex**: one global minimum, but no closed form → iterate (GD, Newton/IRLS, L-BFGS).
- Multinomial: `P = softmax(XW + b)`, `∇_W L = (1/n) Xᵀ(P − Y)` with one-hot `Y`.
- L2 penalty `(λ/2)‖w‖²` adds `λw` to the gradient.
- Decision boundary: `w·x + b = 0`, a hyperplane.

## Complexity
O(n·d) per gradient step (O(n·d·K) for K classes). Prediction O(d). Newton's method costs
O(n·d² + d³) per step but needs very few steps.

## Common pitfalls
- **Perfectly separable data + no regularisation:** weights grow forever (the likelihood keeps
  improving as `‖w‖ → ∞`). Always use some L2 or early stopping.
- Using squared error with a sigmoid: non-convex and gives vanishing gradients when confidently wrong.
- Computing `log(sigmoid(z))` naively → `log(0)`. Use the logits formulation.
- Unscaled features → slow convergence and uneven regularisation.
- Reading accuracy at threshold 0.5 on imbalanced data; choose the threshold from ROC/PR curves.
- Softmax weights are not unique without regularisation (adding a constant vector to every
  class changes nothing).

## When to use vs. alternatives
- Strong, fast, interpretable baseline; gives calibrated probabilities; scales to huge sparse
  data (text, ads).
- Nonlinear boundaries → feature engineering, kernels, trees, neural nets.
- Max-margin with few support vectors → linear SVM (hinge loss). Generative alternative →
  naive Bayes / LDA (better with very little data).

## Interview questions
1. **Why not use linear regression for classification?** Outputs are unbounded, squared error is
   the wrong likelihood, and outliers far on the correct side still shift the boundary.
2. **Derive the gradient.** With `p = σ(z)`, `∂L/∂z = p − y`, so `∇_w = Xᵀ(p − y)/n`.
3. **Is the loss convex? Is there a closed form?** Convex, yes; closed form, no.
4. **What happens on linearly separable data?** MLE does not exist: weights diverge; regularise.
5. **How do you interpret a coefficient?** A one-unit increase in the feature multiplies the
   odds of the positive class by `e^w`, holding the others fixed.
