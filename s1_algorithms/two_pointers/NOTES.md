# Two pointers

## Intuition
Keep two indices into the same sequence and move them according to a rule that lets you
*throw away* candidates without checking them. Each pointer only moves forward (or only
inward), so the total work is O(n) instead of the O(n²) of trying every pair.

Three flavours:
- **Opposite ends** moving inward (sorted pair sum, palindrome, container, rain water).
- **Slow / fast** in the same direction (in-place dedupe, partitioning, cycle detection).
- **Three-way partition** (Dutch national flag).

## Core idea
The proof is always an *invariant*: "the answer is never in the region I skipped".
For sorted two-sum with pointers `i < j`: if `a[i] + a[j] < target`, then `a[i]` paired with
*anything* still in range is too small (since `a[j]` is the largest), so `i` can be discarded.

## Complexity
| Problem | Time | Extra space |
|---|---|---|
| palindrome, two_sum_sorted, container, dedupe, sort_colors, rain water | O(n) | O(1) |
| three_sum | O(n²) (sort + n two-pointer passes) | O(1) besides output |

## Common pitfalls
- Using opposite-end pointers on **unsorted** data (the discard argument needs order).
- `three_sum`: forgetting to skip duplicates at *both* the outer loop and the inner pointer.
- `sort_colors`: advancing `mid` after swapping with `hi` (the swapped-in value is unchecked).
- Off-by-one: `while i < j` vs `while i <= j` (use `<=` only when a single element still matters).

## When to use vs. alternatives
- Sorted input or you can afford to sort, and need O(1) extra space: two pointers.
- Unsorted and need original indices: hash map (see `arrays_hashing`).
- Contiguous subarray with a constraint: sliding window (a same-direction two-pointer).

## Interview questions
1. **Why is it safe to move the shorter wall in the container problem?** Any container using the
   shorter wall and a closer partner is narrower and no taller, so it cannot be better.
2. **How does three-sum avoid duplicates without a set?** Sort, then skip equal neighbours for
   the fixed element and for the left pointer after a hit.
3. **Two-sum: hash map or two pointers?** Hash map is O(n) on unsorted input with O(n) space;
   two pointers is O(1) space but needs sorted input (O(n log n) if you must sort).
4. **What is the invariant in Dutch national flag?** `[0,lo)` zeros, `[lo,mid)` ones,
   `(hi,n)` twos, `[mid,hi]` unknown; the loop shrinks the unknown region to nothing.
5. **How does the O(1)-space rain water solution work?** Process the side with the smaller
   boundary; its water level is fixed by its own running max because the other side is at least as tall.
