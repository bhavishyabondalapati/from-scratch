# Gradient descent variants

## Intuition
You are on a foggy hillside and want to reach the valley. You cannot see far, but you can feel
the slope under your feet. The **gradient** points uphill, so step the other way. Repeat.

- **Batch GD:** measure the slope using *all* the data. Accurate but slow per step.
- **SGD / mini-batch:** estimate the slope from a few random samples. Noisy but cheap, so you
  take many more steps for the same cost; the noise even helps escape shallow traps.
- **Momentum:** a heavy ball. It builds speed where the slope keeps pointing the same way and
  cancels out zig-zagging across a narrow valley.
- **RMSProp:** give each parameter its own step size: small where gradients are large, large
  where they are small.
- **Adam:** momentum + RMSProp + a correction for starting the averages at zero. The default
  choice in deep learning.

## Core math
```
GD        x ← x − η g
Momentum  v ← μ v + g ;              x ← x − η v
RMSProp   s ← ρ s + (1−ρ) g² ;       x ← x − η g / (√s + ε)
Adam      m ← β₁ m + (1−β₁) g ;      v ← β₂ v + (1−β₂) g²
          m̂ = m / (1−β₁ᵗ) ;          v̂ = v / (1−β₂ᵗ)
          x ← x − η m̂ / (√v̂ + ε)
```
- For a quadratic with largest curvature L, GD converges iff `η < 2/L`.
- Convergence speed depends on the **condition number** κ = L/μ: GD needs O(κ) steps,
  momentum O(√κ).
- Mini-batch gradient is an unbiased estimate of the full gradient; its variance shrinks like
  1/batch_size.

## Complexity
Per step: O(cost of one gradient) + O(d) for the update. Memory: SGD 0 extra, momentum and
RMSProp d, Adam 2d (d = number of parameters).

## Common pitfalls
- Learning rate too high → divergence/NaN; too low → looks like "no learning".
- Unscaled features make the loss surface a long narrow valley (bad κ): standardise inputs.
- Forgetting to shuffle each epoch (ordered data biases SGD).
- Different libraries define momentum/RMSProp slightly differently (dampening, ε inside or
  outside the square root) — matters when reproducing results.
- Adam + L2 penalty is not the same as weight decay; use AdamW (decoupled decay).
- Convex guarantees do not apply to neural nets: you get *a* good minimum, not *the* minimum.

## When to use vs. alternatives
- Adam/AdamW: default for deep nets, transformers, sparse gradients; little tuning.
- SGD + momentum: often generalises better for vision models with a tuned schedule.
- Small problems with exact solutions (linear regression) → closed form / Newton / L-BFGS.
- Second-order methods (Newton) use curvature and converge in few steps but cost O(d³).

## Interview questions
1. **Batch vs stochastic vs mini-batch?** Trade gradient accuracy for cost per step; mini-batch
   is the practical middle and uses vectorised hardware well.
2. **Why does momentum help?** It averages gradients, damping oscillation across steep
   directions and accelerating along consistent ones.
3. **Why does Adam need bias correction?** `m` and `v` start at 0, so early averages are too
   small by a factor `(1 − βᵗ)`; dividing by it fixes the first steps.
4. **What happens if the learning rate is too large?** Updates overshoot; on a quadratic the
   error grows whenever `η > 2/L`.
5. **Adam vs SGD?** Adam adapts per-parameter step sizes and converges fast with defaults; SGD
   with momentum can reach flatter minima and better test accuracy when tuned.
