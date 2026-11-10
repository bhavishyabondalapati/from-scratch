# Fenwick tree (binary indexed tree)

## Intuition
A plain array makes updates O(1) but prefix sums O(n). A prefix-sum array makes prefix sums O(1)
but updates O(n). A Fenwick tree meets in the middle: both are O(log n), in about ten lines.

Every number is a sum of powers of two (13 = 8 + 4 + 1), so every prefix of length 13 can be cut
into blocks of length 8, 4 and 1. The tree stores exactly those power-of-two-sized block sums.

## Core math
- `lowbit(i) = i & -i` = value of the lowest set bit (two's complement trick).
- `tree[i]` = sum of the elements `(i - lowbit(i), i]` (1-indexed).
- **Query** prefix `i`: add `tree[i]`, then `i -= lowbit(i)`, until 0.
- **Update** position `i`: add to `tree[i]`, then `i += lowbit(i)`, until past n.
- Range sum = `prefix(hi) - prefix(lo)`, which is why the operation must be invertible.

## Complexity
| Operation | Time | Space |
|---|---|---|
| Build | O(n) | n + 1 |
| add, prefix_sum, range_sum | O(log n) | |
| find_kth (binary lifting) | O(log n) | |
| count_inversions | O(n log n) | O(n) |

## Common pitfalls
- 0-indexing the internal array: `i & -i` is 0 for `i = 0`, causing an infinite loop.
- Using it for min/max: you cannot "subtract" a prefix min (only works for prefix-only queries
  with monotone updates).
- `add` adds a delta; to *set* a value, add `new - old`.
- For values that are large or negative, coordinate-compress to ranks first.

## When to use vs. alternatives
- Point update + range sum, counting/order statistics, inversions → Fenwick (smallest, fastest).
- Range min/max, non-invertible ops, complex lazy updates → segment tree.
- No updates → prefix sums.

## Interview questions
1. **What does `i & -i` compute and why?** The lowest set bit: `-i` is `~i + 1`, which flips all
   bits above the lowest set bit and preserves that bit.
2. **Why are both operations O(log n)?** Query removes one set bit per step; update moves to a
   strictly higher power-of-two block per step; both are bounded by the number of bits.
3. **How do you count inversions with it?** Process elements in order; for each, count
   previously seen elements with greater rank, then insert its rank.
4. **How do you support range updates?** Store the difference array (range add / point query),
   or two trees for range add / range sum.
5. **Fenwick vs segment tree?** Same asymptotics for sums; Fenwick uses n memory and trivial
   code, but only supports invertible operations.
