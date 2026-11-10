# Scaled dot-product attention and multi-head attention

## Intuition
Attention lets every position in a sequence **look at every other position and decide how much to
borrow from each**. Each token produces three vectors:
- a **query** — "what am I looking for?"
- a **key** — "what do I contain?"
- a **value** — "what will I hand over if chosen?"

A token's output is the average of all values, weighted by how well its query matches each key
(softmax of the dot products). Unlike an RNN, any two positions are connected in one step, and all
positions are computed in parallel.

**Multi-head** attention runs several smaller attentions side by side, each with its own query/key/
value projections, so different heads can track different relationships (the previous word, the
subject, a matching bracket...), then concatenates and mixes their outputs.

## Core math
- `Attention(Q, K, V) = softmax(QKᵀ / √d_k + M) V`, `M` = 0 or −∞ (masking).
- Why `√d_k`: for random unit-variance vectors, `q·k` has variance `d_k`. At d = 512 the raw scores
  have std ≈ 22 (softmax almost one-hot, tiny gradients); scaled, std ≈ 1 (tested).
- Causal mask: position i may attend only to j ≤ i (language modelling).
- Multi-head: `headᵢ = Attention(XW_Qⁱ, XW_Kⁱ, XW_Vⁱ)`, `MHA = Concat(head₁..h) W_O`, `d_head = d/h`.
- Self-attention without positions is **permutation-equivariant** (tested): it has no notion of
  order until positional encodings are added.
- Matches `F.scaled_dot_product_attention` and `nn.MultiheadAttention` (weights copied) to float
  tolerance, including causal masks and key-padding masks; gradcheck passes.

## Complexity
O(T² · d) time and O(T²) memory per head for the score matrix: quadratic in sequence length.
FlashAttention computes the same result in tiles without storing the T×T matrix.

## Common pitfalls
- Mask conventions differ: here `True` = attend; `nn.MultiheadAttention` uses `True` = block.
- Fully masked rows: softmax of all −∞ is NaN; handle them (zeros here).
- Forgetting the scale factor; applying the mask after the softmax (weights no longer sum to 1).
- Shape bugs when splitting heads: view as (B, T, h, d_head) then transpose to (B, h, T, d_head).
- Padding tokens must be masked as keys (and ignored in the loss).

## When to use vs. alternatives
- The core of transformers (`gpt`, `bert_classifier`), cross-attention in seq2seq, vision transformers.
- Very long sequences: sparse/local attention, linear attention, state-space models.
- Additive (Bahdanau) attention is the older variant used in `seq2seq_attention`.

## Interview questions
1. **Why divide by √d_k?** To keep score variance ≈ 1 so the softmax does not saturate and
   gradients do not vanish.
2. **What are Q, K, V?** Learned projections of the inputs: queries are matched against keys, and the
   matching weights average the values.
3. **Why multiple heads?** Each head attends in a different subspace, letting the layer capture
   several relationships at once for the same cost.
4. **What is the complexity of self-attention?** O(T²·d) time, O(T²) memory.
5. **What does the causal mask do?** Prevents positions from attending to the future so a model can be
   trained to predict the next token in parallel.
