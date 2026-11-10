# Sequence DP: LIS, LCS, edit distance

## Intuition
- **LIS (longest increasing subsequence):** pick elements, keeping their order, so they strictly
  increase; make it as long as possible.
- **LCS (longest common subsequence):** the longest sequence appearing (in order, not necessarily
  contiguously) in *both* inputs. This is what `diff` computes.
- **Edit distance:** the fewest single-character insertions, deletions and substitutions to turn
  one string into another. This powers spell-checkers and DNA alignment.

LCS and edit distance share one shape: a 2D table over *prefixes* of the two strings, where each
cell looks at the last characters and at three neighbours (up, left, diagonal).

## Core math
- LIS O(n²): `best[i] = 1 + max(best[j] for j < i if a[j] < a[i])`.
- LIS O(n log n): `tails[k]` = smallest tail of any increasing subsequence of length k + 1.
  `tails` is sorted, so each element is placed by binary search (replace the first tail ≥ x).
  `tails` is **not** itself an LIS; only its length is meaningful.
- LCS: `L[i][j] = L[i-1][j-1] + 1` if `a[i-1] == b[j-1]`, else `max(L[i-1][j], L[i][j-1])`.
- Edit distance: `D[i][j] = D[i-1][j-1]` if equal, else `1 + min(D[i-1][j], D[i][j-1], D[i-1][j-1])`;
  base cases `D[i][0] = i`, `D[0][j] = j`.

## Complexity
| Problem | Time | Space |
|---|---|---|
| LIS (patience) | O(n log n) | O(n) |
| LIS (classic DP) | O(n²) | O(n) |
| LCS, edit distance | O(n · m) | O(n · m), or O(min(n, m)) for the length only |

## Common pitfalls
- Subsequence (gaps allowed) vs substring (contiguous): different recurrences.
- Strict vs non-decreasing LIS: `bisect_left` gives strict, `bisect_right` gives non-decreasing.
- Returning `tails` as the answer sequence.
- Edit distance base row/column of zeros instead of 0..n.
- Space-optimising to one row and then trying to reconstruct the path (you need the full table,
  or Hirschberg's algorithm).

## When to use vs. alternatives
- Similarity of two sequences where order matters → LCS / edit distance.
- Only substitutions allowed, equal lengths → Hamming distance (O(n)).
- Very long strings with few differences → Myers' O(ND) diff or banded DP.
- LIS also solves "minimum number of removals to sort", box/envelope nesting (sort, then LIS).

## Interview questions
1. **Explain the O(n log n) LIS.** Maintain the smallest possible tail for each length; binary
   search where each new element goes. The array's length is the answer.
2. **Why is `tails` always sorted?** A longer increasing subsequence's tail must exceed the tail
   of its own prefix, which is at least the smallest tail of that shorter length.
3. **How are LCS and edit distance related?** With only insert/delete allowed,
   `distance = len(a) + len(b) - 2·LCS(a, b)`.
4. **How do you reconstruct the edits?** Walk back from `D[n][m]`, at each cell moving to the
   neighbour that explains the value.
5. **How do you reduce space to O(min(n, m))?** Keep only the previous row; put the shorter
   string on the column axis.
