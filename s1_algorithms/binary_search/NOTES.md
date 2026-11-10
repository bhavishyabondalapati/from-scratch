# Binary search

## Intuition
Binary search is not really about sorted arrays. It is about a **yes/no question whose answer
flips exactly once** as you move along a range: `No No No Yes Yes Yes`. Ask the question in the
middle; whichever answer you get, half the range is ruled out. "Is `nums[mid] >= target`?" is one
such question; "can we finish in time at speed `mid`?" is another (search on the *answer space*).

## Core idea
Maintain the invariant "the boundary is inside `[lo, hi]`". Template used here:

```
while lo < hi:
    mid = (lo + hi) // 2
    if pred(mid): hi = mid        # mid could be the first True
    else:         lo = mid + 1    # mid is False, boundary is to the right
return lo
```
The range shrinks every iteration because `mid < hi` whenever `lo < hi`.

## Complexity
O(log n) probes. For answer-space search: O(log(range) × cost of one feasibility check), e.g.
`min_eating_speed` is O(n log max(piles)). Space O(1).

## Common pitfalls
- Infinite loop from `lo = mid` with `mid = (lo + hi) // 2` (rounds down, never advances).
  If you need `lo = mid`, round up: `mid = (lo + hi + 1) // 2`.
- Mixing the two styles: `while lo <= hi` goes with `hi = mid - 1`; `while lo < hi` goes with `hi = mid`.
- Overflow of `lo + hi` in fixed-width languages (use `lo + (hi - lo) // 2`); not an issue in Python.
- Answer-space search with a predicate that is **not monotonic**.
- Rotated arrays with duplicates: `nums[lo] == nums[mid]` is ambiguous; worst case degrades to O(n).

## When to use vs. alternatives
- Static sorted data, lookups → binary search (O(log n), O(1) space).
- Frequent inserts/deletes → balanced BST / hash map.
- "Minimise the maximum" / "maximise the minimum" with a cheap greedy check → search the answer.
- Unimodal function (single peak) → ternary search or binary search on the slope.

## Interview questions
1. **lower_bound vs upper_bound?** First index with value `>= target` vs `> target`; their
   difference is the count of `target`.
2. **How do you recognise binary search on the answer?** You are asked to optimise a number, a
   candidate answer can be verified quickly, and feasibility is monotonic.
3. **Why does `while lo < hi` terminate?** `mid` is always `< hi`, so `hi = mid` shrinks the
   range, and `lo = mid + 1` shrinks it too.
4. **How does rotated-array search decide which half to keep?** One half is always sorted; check
   whether the target lies inside that sorted half's endpoints.
5. **What breaks with duplicates in a rotated array?** You cannot tell which half is sorted when
   the ends equal the middle, so you must shrink linearly in the worst case.
