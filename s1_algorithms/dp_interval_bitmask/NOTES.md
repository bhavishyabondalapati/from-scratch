# Interval DP and bitmask DP

## Intuition
**Interval DP.** The state is a contiguous range `[i, j]` of the input. The answer for a range is
found by trying every way to split it into two smaller ranges (or every choice of the *last*
action inside it). Solve ranges in order of increasing length so the pieces are ready.

**Bitmask DP.** Sometimes the state must remember *which items have been used*, not just how
many. For small n, a set of items fits in the bits of one integer: bit `i` is 1 iff item `i` is
used. The DP table is indexed by that integer. This turns n! orderings into 2ⁿ subsets.

## Core math
- Matrix chain: `cost[i][j] = min_k cost[i][k] + cost[k+1][j] + d[i]·d[k+1]·d[j+1]`.
- Burst balloons: choose the **last** balloon `k` in the open interval `(i, j)`:
  `best[i][j] = max_k best[i][k] + best[k][j] + v[i]·v[k]·v[j]`.
- Palindromes: `pal[i][j] = s[i] == s[j] and pal[i+1][j-1]`.
- Held-Karp TSP: `best[mask][last] = min over prev in mask: best[mask − {last}][prev] + d[prev][last]`.
- Bit tricks: test `mask & (1 << i)`, set `mask | (1 << i)`, clear `mask & ~(1 << i)`,
  full set `(1 << n) − 1`, count `bin(mask).count("1")`, iterate submasks `sub = (sub − 1) & mask`.

## Complexity
| Problem | Time | Space |
|---|---|---|
| Matrix chain, burst balloons | O(n³) | O(n²) |
| Palindrome partitioning | O(n²) | O(n²) |
| Longest palindromic substring (expand) | O(n²) | O(1) |
| TSP (Held-Karp) | O(n² · 2ⁿ) | O(n · 2ⁿ) |
| Assignment | O(n · 2ⁿ) | O(2ⁿ) |
| Partition into k equal subsets | O(n · 2ⁿ) | O(2ⁿ) |

## Common pitfalls
- Interval DP: iterating `i` then `j` in increasing order uses ranges that are not computed
  yet; iterate by **length** (or `i` descending).
- Burst balloons: thinking about the *first* balloon burst makes subproblems depend on each other.
- Bitmask: operator precedence (`mask & 1 << i == 0` is wrong; parenthesise).
- 2ⁿ blows up quickly: n = 20 is ~1M states (fine), n = 30 is ~1B (not).
- Storing redundant state: often something (e.g. "which worker is next") is implied by the mask.

## When to use vs. alternatives
- "Merge/split a sequence optimally", parenthesisation, palindromes → interval DP.
- "Visit/assign every item exactly once", n ≤ ~20 → bitmask DP.
- Assignment for large n → Hungarian algorithm (O(n³)). TSP for large n → heuristics / approximation.

## Interview questions
1. **Why order interval DP by length?** `[i, j]` depends on strictly shorter ranges inside it.
2. **Why "last balloon" rather than "first balloon"?** The last balloon in a range has fixed
   neighbours (the range boundaries), which decouples the left and right subranges.
3. **What does Held-Karp save over brute force?** It forgets the *order* of visited cities and
   keeps only the set and the current city: O(n² 2ⁿ) vs O(n!).
4. **How do you represent and update a set in a bitmask?** Bit i for item i; OR to add, AND with
   the complement to remove, AND to test.
5. **Largest n for bitmask DP?** Roughly 20–22 for O(n · 2ⁿ) within a second or two in a fast
   language; less in Python.
