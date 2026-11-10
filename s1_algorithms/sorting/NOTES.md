# Sorting algorithms

## Intuition
- **Insertion sort:** sort a hand of cards — pick up one card at a time and slide it into place.
- **Merge sort:** split the pile in two, sort each half, then zip the two sorted halves together.
- **Quicksort:** pick a pivot, put smaller things left and bigger things right, repeat on each side.
- **Heap sort:** build a max-heap, then keep moving the maximum to the end.
- **Counting sort:** if values are small integers, just count how many of each there are.
- **Radix sort:** sort by the last digit, then the next digit, and so on, using a *stable* sort.

## Core math
- Comparison sorts need Ω(n log n) comparisons: there are n! possible orderings and each
  comparison at best halves the candidates, so you need log₂(n!) ≈ n log₂ n comparisons.
- Counting/radix sort beat this bound because they do not compare; they index by value.
- Merge sort recurrence: T(n) = 2T(n/2) + O(n) = O(n log n).
- Quicksort: balanced pivots give the same recurrence; a worst pivot every time gives
  T(n) = T(n−1) + O(n) = O(n²). A random pivot makes the expected cost O(n log n) on any input.

## Complexity
| Algorithm | Best | Average | Worst | Extra space | Stable |
|---|---|---|---|---|---|
| Insertion | O(n) | O(n²) | O(n²) | O(1) | yes |
| Merge | O(n log n) | O(n log n) | O(n log n) | O(n) | yes |
| Quick (random pivot, 3-way) | O(n) (all equal) | O(n log n) | O(n²) | O(log n) | no |
| Heap | O(n log n) | O(n log n) | O(n log n) | O(1) | no |
| Counting | O(n + k) | O(n + k) | O(n + k) | O(n + k) | yes |
| Radix (LSD) | O(d(n + b)) | same | same | O(n + b) | yes |

`k` = value range, `d` = number of digits, `b` = base.

## Common pitfalls
- Quicksort with the first element as pivot on sorted input → O(n²) and deep recursion.
- Two-way partition with many duplicates → O(n²); use three-way partitioning.
- Merge using `<` instead of `<=` loses stability.
- Counting sort when the range is huge (k ≫ n) wastes memory.
- Radix sort with an unstable inner sort gives wrong results; negatives need special handling.

## When to use vs. alternatives
- General purpose: use the library sort (Python's Timsort = merge + insertion, stable,
  O(n) on already-sorted runs).
- Need guaranteed O(n log n) and O(1) space → heap sort. Need stability / linked lists /
  external (on-disk) sorting → merge sort. Best average speed in place → quicksort.
- Small integer keys → counting sort; fixed-width integers or strings → radix sort.
- Only the k smallest → heap or quickselect, not a full sort.

## Interview questions
1. **Why is comparison sorting Ω(n log n)?** A decision tree distinguishing n! permutations has
   height at least log₂(n!) = Θ(n log n).
2. **What is stability and when does it matter?** Equal keys keep their original order; needed
   for multi-key sorts (sort by name, then stably by age) and for radix sort passes.
3. **Quicksort's worst case and how to avoid it?** Consistently unbalanced pivots; use random or
   median-of-three pivots, 3-way partitioning, and fall back to heap sort (introsort).
4. **Merge sort vs quicksort?** Merge: stable, guaranteed bound, O(n) memory. Quick: in place,
   cache-friendly, faster on average, not stable.
5. **What does Python's `sorted` use?** Timsort: finds existing runs, extends short ones with
   insertion sort, merges them; stable and adaptive.
