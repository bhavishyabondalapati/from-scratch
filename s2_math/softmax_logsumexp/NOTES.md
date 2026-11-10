# Softmax / log-sum-exp stability

## Intuition
**Softmax** turns a list of arbitrary scores ("logits") into probabilities: exponentiate so
everything is positive, then divide by the total so it sums to 1. Bigger score → bigger share.

The catch: computers store numbers with limited range. `exp(1000)` is "infinity" in float64, and
`inf / inf` is `nan`. `exp(-1000)` is exactly `0`, and `log(0)` is `-inf`. A model that was
training fine suddenly prints `nan` — this is very often the reason.

The fixes are small algebra tricks that give the *same* math with *safe* intermediate numbers.

## Core math
- `softmax(z)ᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}`. Adding a constant to all logits changes nothing, so
  subtract `m = max(z)`: every exponent is ≤ 0 and at least one term is `e⁰ = 1`.
- `logsumexp(z) = m + log Σ e^{zᵢ − m}`. It is a smooth version of `max`:
  `max(z) ≤ LSE(z) ≤ max(z) + log n`.
- `log_softmax(z) = z − logsumexp(z)` (never compute `log(softmax(z))` in two steps).
- Cross-entropy from logits: `L = −log_softmax(z)[y]`; gradient `∂L/∂z = softmax(z) − onehot(y)`.
- `∇ logsumexp(z) = softmax(z)`.
- Stable sigmoid: `1/(1+e^{−x})` for `x ≥ 0`, `e^{x}/(1+e^{x})` for `x < 0`.
- BCE from logits: `max(x, 0) − x·y + log(1 + e^{−|x|})`.
- `log1p(t)` computes `log(1 + t)` accurately for tiny `t`.

## Complexity
O(n) per vector; the stable forms cost one extra `max` pass.

## Common pitfalls
- Applying softmax and then a loss that expects logits (PyTorch `CrossEntropyLoss` does
  log-softmax internally): double softmax silently hurts training.
- `log(softmax(z))` or `log(sigmoid(x))` in two steps → `-inf` and then `nan` gradients.
- Clipping probabilities (`p.clip(1e-12)`) hides the problem and biases the loss; work in log space.
- Wrong `axis` on batched inputs (normalising across the batch instead of across classes).
- Mixed precision / float32 overflows much earlier (`exp(89)`).

## When to use vs. alternatives
- Multi-class, exactly one label → softmax + cross-entropy.
- Multi-label (independent yes/no per class) → sigmoid + BCE per class.
- Any sum of probabilities in log space (HMM forward algorithm, mixture likelihoods, CTC,
  attention masks with `-inf`) → logsumexp.
- Temperature `softmax(z / T)`: T < 1 sharpens, T > 1 flattens (used in sampling, distillation).

## Interview questions
1. **How do you make softmax numerically stable?** Subtract the max logit before exponentiating;
   the result is mathematically identical.
2. **Why do frameworks combine log-softmax and NLL into one loss?** To use the log-sum-exp trick
   and avoid `log(0)`; it also gives the simple gradient `p − y`.
3. **What is the gradient of cross-entropy w.r.t. the logits?** `softmax(z) − onehot(y)`.
4. **What is logsumexp intuitively?** A smooth, differentiable approximation of `max` whose
   gradient is the softmax.
5. **Softmax vs sigmoid outputs?** Softmax: classes compete, probabilities sum to 1. Sigmoid:
   independent probabilities per class. For two classes, softmax over `[z, 0]` equals `sigmoid(z)`.
