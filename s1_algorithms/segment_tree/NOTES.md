# Segment tree

## Intuition
Prefix sums answer range queries instantly but break when the array changes. A segment tree
stores the answer for a set of **pre-chosen ranges** arranged as a binary tree: the root covers
everything, its children cover each half, and so on down to single elements. Any range can be
stitched together from at most ~2·log n of those stored pieces, and changing one element only
invalidates the log n pieces that contain it.

## Core idea
- Node = combined value of its segment; `node = op(left child, right child)`.
- Works for any **associative** `op` with an identity: sum, min, max, gcd, XOR, matrix product.
- **Lazy propagation** (for range *updates*): when an update fully covers a node, update the
  node's own value and leave a note (`lazy`) for the children; push the note down only when a
  later operation needs to go below that node.

## Complexity
| Operation | Time | Space |
|---|---|---|
| Build | O(n) | 2n (iterative) / 4n (recursive) |
| Point update, range query | O(log n) | |
| Range update with lazy | O(log n) | |

## Common pitfalls
- Allocating 2n for the *recursive* layout when n is not a power of two (needs up to 4n).
- Wrong identity element (e.g. 0 for min).
- Lazy: forgetting to push down before recursing, or to recompute the parent afterwards.
- Lazy sum: the pending add must be multiplied by the segment length.
- Combining left and right partial results in the wrong order for non-commutative ops.
- Mixing inclusive and exclusive bounds.

## When to use vs. alternatives
- Static array, sums → prefix sums (O(1) query).
- Point update + prefix/range **sum** → Fenwick tree (same big-O, far less code and memory).
- Range min/max with updates, range updates, or custom merges → segment tree.
- Static range min → sparse table (O(1) query).

## Interview questions
1. **Why is a query O(log n)?** At each level at most two nodes are partially overlapped; all
   others are fully inside (used whole) or outside (skipped).
2. **What property must `op` have?** Associativity, so partial results can be grouped freely;
   commutativity is not required if you keep left/right order.
3. **What is lazy propagation?** Deferring a range update by tagging the highest fully covered
   nodes and pushing tags down only on demand.
4. **Segment tree vs Fenwick tree?** Fenwick is simpler/faster for invertible ops like sum;
   segment trees handle min/max and richer range updates.
5. **How would you support "range assign" as well as "range add"?** Keep a richer lazy tag and
   define how tags compose (assign overrides earlier adds).
