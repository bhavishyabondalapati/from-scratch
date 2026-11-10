# Numerical gradient checking

## Intuition
A derivative is "how much does the output change if I nudge the input a tiny bit?". So you can
*measure* a gradient without any calculus: nudge each input up and down by a small ε, see how
the output changes, divide. It is far too slow for training (one or two function calls **per
parameter**), but it is the gold standard for **testing** a hand-written backward pass.

If backprop says the gradient is `g` and the measured slope disagrees, backprop has a bug.

## Core math
- Forward difference: `(f(x+ε) − f(x)) / ε = f'(x) + O(ε)`.
- Central difference: `(f(x+ε) − f(x−ε)) / 2ε = f'(x) + O(ε²)` (the odd Taylor terms cancel).
- Two error sources pull in opposite directions:
  - truncation error ∝ ε² (shrinks as ε shrinks),
  - floating-point round-off ∝ machine_eps / ε (grows as ε shrinks).
  The sweet spot for float64 central differences is ε ≈ 1e-5 … 1e-6.
- Relative error: `|a − n| / (|a| + |n|)`. Rough guide in float64: < 1e-7 correct; 1e-4…1e-2
  suspicious; > 1e-2 wrong.

## Complexity
`2 · (number of parameters)` function evaluations for one gradient: O(P · cost(f)).
Backprop gets the same gradient in about the cost of **one** extra evaluation, which is why
numerical gradients are only for checking.

## Common pitfalls
- Using float32: round-off dominates and correct gradients look wrong. Check in float64.
- Non-differentiable points (ReLU at 0, max-pool ties) give misleading differences; nudge the
  inputs away from kinks.
- Randomness inside `f` (dropout, sampling): fix the seed/mask so both evaluations match.
- Forgetting a term of the loss (e.g. the regulariser) in either the analytic or numeric side.
- Checking only at initialisation where some paths are inactive; also check after a few steps.
- Comparing with absolute error: tiny gradients trivially "match".
- Integer arrays: adding 1e-5 to an int does nothing.

## When to use vs. alternatives
- Unit-testing any custom backward pass, layer, or loss (used throughout sections 3–5 here).
- Automatic differentiation (PyTorch/JAX) is exact and fast: use it for real training, and use
  finite differences to test *it* when you write custom autograd functions.
- Complex-step differentiation avoids cancellation entirely when `f` supports complex inputs.

## Interview questions
1. **Why central rather than forward differences?** Error O(ε²) instead of O(ε) for one extra
   function call.
2. **Why not make ε extremely small?** Subtracting nearly equal floats loses precision
   (catastrophic cancellation); error grows like machine_eps/ε.
3. **Why relative error?** It is independent of the gradient's scale.
4. **Why not train with numerical gradients?** Cost is proportional to the number of
   parameters per step; backprop is proportional to one forward pass.
5. **Your gradient check fails only for some units. What do you look at?** Kinks (ReLU at 0),
   a stochastic layer not being frozen, float32, or a broadcasting bug that sums over the
   wrong axis.
