# Prefix sums

## Intuition
If you know the running total up to every position, the total of any stretch in the middle is
"total up to the end of the stretch" minus "total up to its start". Pay O(n) once, then answer
every range-sum question in O(1).

## Core math
- `prefix[0] = 0`, `prefix[i] = nums[0] + ... + nums[i-1]`.
- `sum(nums[lo:hi]) = prefix[hi] - prefix[lo]`.
- 2D (inclusion-exclusion): `S = P[r2][c2] - P[r1][c2] - P[r2][c1] + P[r1][c1]`.
- **Difference array** is the inverse: store `d[i] = a[i] - a[i-1]`; a range add becomes two
  point updates, and a prefix sum of `d` recovers `a`.
- **Prefix + hash map:** `sum(i..j) = k` iff `prefix[i] = prefix[j] - k`, turning "count
  subarrays" into "count earlier prefixes with a given value".

## Complexity
| Operation | Time | Space |
|---|---|---|
| Build 1D / 2D | O(n) / O(rows·cols) | same |
| Range query | O(1) | |
| subarray_sum_equals_k, longest_subarray_with_sum | O(n) | O(n) |
| m range updates via difference array | O(n + m) | O(n) |

## Common pitfalls
- Off-by-one: using a prefix array of length n instead of n + 1 (the leading 0 removes all edge cases).
- Forgetting to seed the hash map with the empty prefix (`{0: 1}` or `{0: -1}`).
- For "longest", overwriting the stored index instead of keeping the earliest.
- Prefix sums are for **static** arrays; any update costs O(n) to repair.

## When to use vs. alternatives
- Static array, many range sums → prefix sums.
- Updates mixed with queries → Fenwick tree or segment tree (O(log n) both).
- Range min/max (not invertible, cannot "subtract") → sparse table or segment tree.
- All-positive numbers with a sum constraint → sliding window uses O(1) space.

## Interview questions
1. **Why does subarray-sum-equals-k need a hash map rather than a sliding window?** With negative
   numbers the window sum is not monotonic, so shrinking/growing decisions are not safe.
2. **Why length n + 1?** So that ranges starting at index 0 need no special case.
3. **How do you apply 10⁵ range increments to an array efficiently?** Difference array: two
   writes per update, then one prefix-sum pass.
4. **Can prefix sums answer range minimum?** No: min has no inverse operation. Sum, XOR and
   (non-zero) product do.
5. **Derive the 2D formula.** Add the big rectangle, subtract the strip above and the strip to the
   left, add back the corner that was subtracted twice.
