# Quickselect

## Intuition
To find the k-th smallest element you do not need the whole array sorted; you only need the
element that *would* land at position k. Partition around a pivot as in quicksort: the pivot ends
up at its final sorted position. If that position is k, done. Otherwise the answer is entirely on
one side, so **throw the other side away** and repeat.

## Core math
- Average case with a random pivot: each round keeps about half → n + n/2 + n/4 + … ≤ 2n = O(n).
- Worst case (always the smallest/largest pivot): n + (n−1) + … = O(n²).
- **Median of medians:** split into groups of 5, take each group's median, recursively take the
  median of those. That pivot is ≥ about 3n/10 elements and ≤ about 3n/10, so
  T(n) ≤ T(n/5) + T(7n/10) + O(n). Since 1/5 + 7/10 < 1, T(n) = O(n) guaranteed.

## Complexity
| Method | Time | Space |
|---|---|---|
| Sort then index | O(n log n) | O(1)–O(n) |
| Heap of size k | O(n log k) | O(k) |
| Quickselect (random pivot) | O(n) average, O(n²) worst | O(1) in place (O(n) here for the copy) |
| Median of medians | O(n) worst case (large constant) | O(n) |

## Common pitfalls
- Off-by-one between "k-th smallest (1-based)", "index k (0-based)" and "k-th largest".
- Fixed pivot (first/last element) → O(n²) on sorted input.
- Two-way partition with many duplicates → O(n²); the 3-way partition here returns as soon as
  k lands in the "equal" block.
- Quickselect reorders the array (this implementation copies first so callers are not surprised).
- Not suitable for streams: needs all data in memory (use a heap).

## When to use vs. alternatives
- One-off k-th element / median / top-k on in-memory data → quickselect (`numpy.partition`,
  C++ `nth_element`).
- Streaming data or tiny k → heap.
- Many different k on the same data → sort once.
- Hard real-time guarantee → median of medians or introselect.

## Interview questions
1. **Why is quickselect O(n) on average while quicksort is O(n log n)?** Quickselect recurses into
   one side only, giving a geometric series instead of n work per level over log n levels.
2. **How do you get a worst-case linear guarantee?** Median-of-medians pivot selection.
3. **Why groups of 5?** It is the smallest odd size where the recurrence fractions sum below 1
   (groups of 3 give T(n/3) + T(2n/3), which is O(n log n)).
4. **Heap vs quickselect for top-k?** Heap: O(n log k), streaming, O(k) memory. Quickselect:
   O(n) average, needs random access to all data.
5. **How do you find the median of an unsorted array in O(n)?** Quickselect for index n//2 (and
   n//2 − 1 for even n).
