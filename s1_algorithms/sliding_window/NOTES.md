# Sliding window

## Intuition
When the question is about a **contiguous** subarray/substring, do not recompute each candidate
from scratch. Keep a window `[left, right]`, extend `right` one step at a time, and move `left`
forward only when the window breaks a rule. Because adding/removing one element updates the
window's summary (sum, counts) in O(1), the whole scan is O(n).

- **Fixed size:** add the new element, subtract the one that fell off.
- **Variable size, "longest valid":** grow right; while invalid, shrink left.
- **Variable size, "shortest valid":** grow right; while valid, record and shrink left.

## Core idea
Works when validity is **monotonic**: if a window is valid, every sub-window is valid (longest
problems), or if it is valid, every super-window is valid (shortest problems). Then `left` never
needs to move backward, so both pointers move at most n times: 2n steps total.

## Complexity
All functions here: O(n) time. Space O(1) for sums, O(alphabet) for character counts.

## Common pitfalls
- Using the sum-based window when numbers can be **negative**: shrinking no longer reliably
  reduces the sum, so monotonicity breaks (use prefix sums + hash map / monotonic deque instead).
- `longest_unique_substring`: jumping `left` backward when the previous occurrence is already
  outside the window (hence the `>= left` check).
- Comparing Counters that still contain zero-count keys.
- Off-by-one in window length: it is `right - left + 1`.

## When to use vs. alternatives
- Contiguous + monotonic validity → sliding window.
- Subarray sum equals k with negatives → prefix sums + hash map.
- Max/min of every window → monotonic deque (see `queue_deque`).
- Non-contiguous subsequence → usually DP.

## Interview questions
1. **Why is a nested `while` inside the `for` still O(n)?** `left` only moves forward and at most
   n times in total across the entire run (amortized analysis).
2. **Why does min-subarray-sum need positive numbers?** So adding an element never decreases the
   sum and removing never increases it; that monotonicity justifies never moving `left` back.
3. **In character replacement, why is a stale `max_count` OK?** The answer only grows when a
   strictly larger true `max_count` appears; a stale value just keeps the window from shrinking
   below the best size already achieved.
4. **How do you track "window contains all of t" in O(1) per step?** Keep a `missing` counter
   of still-needed characters instead of comparing whole count tables.
5. **Fixed vs. variable window?** Fixed when the size is given; variable when you optimise the
   size subject to a constraint.
