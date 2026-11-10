# Dynamic programming: 1D and 2D grid

## Intuition
Dynamic programming (DP) is "recursion without repeated work". If a problem can be answered by
combining answers to smaller versions of itself, and those smaller versions *repeat*, solve each
one once and remember it.

Two ingredients must be present:
- **Optimal substructure:** the best answer is built from best answers to subproblems.
- **Overlapping subproblems:** the same subproblem is needed many times (otherwise plain
  divide-and-conquer suffices).

## Core idea — the five steps
1. **State:** what does `dp[i]` (or `dp[r][c]`) mean, in one sentence?
2. **Recurrence:** how does it depend on smaller states? (Think about the *last* decision.)
3. **Base cases.**
4. **Order:** fill so that dependencies are computed first (or memoize top-down).
5. **Space:** keep only what the recurrence looks back at.

| Problem | State | Recurrence |
|---|---|---|
| house robber | best using first i houses | `max(dp[i-1], dp[i-2] + v[i])` |
| max subarray (Kadane) | best sum ending at i | `max(x, dp[i-1] + x)` |
| coin change | fewest coins for amount a | `1 + min(dp[a-c])` |
| word break | can s[:i] be segmented | `any(dp[j] and s[j:i] in words)` |
| unique paths | #paths to (r, c) | `dp[r-1][c] + dp[r][c-1]` |
| maximal square | side of square ending at (r, c) | `1 + min(up, left, diag)` |

## Complexity
Time = (number of states) × (work per state). Space = number of states you keep.
Most 1D problems here: O(n) time, O(1) space. Coin change: O(amount × coins).
Word break: O(n × L) set lookups. Grids: O(rows × cols) time, O(cols) space possible.

## Common pitfalls
- Vague state definition ("best so far" vs "best *ending here*" are different problems).
- Greedy coin change (largest coin first) fails for coins like {1, 3, 4} and amount 6.
- Off-by-one base cases; using padding (an extra row/column of zeros) removes them.
- Top-down recursion depth in Python for large n.
- Initialising "impossible" with a value that can be mistaken for a real answer.

## When to use vs. alternatives
- Counting ways / min / max / feasibility over sequential decisions → DP.
- Local choice provably optimal → greedy (faster, simpler).
- Need to list all solutions → backtracking.
- Grid shortest path with moves in all four directions → BFS/Dijkstra (DP needs an acyclic order).

## Interview questions
1. **Top-down vs bottom-up?** Memoization is closest to the recurrence and only visits reachable
   states; tabulation avoids recursion overhead and makes space optimisation easy.
2. **How do you find the recurrence?** Ask what the last step of an optimal solution looks like
   and what smaller problem remains.
3. **Why does Kadane work?** A best subarray ending at i either extends the best one ending at
   i − 1 or starts at i; a negative running sum can never help.
4. **Why can unique-paths use one row?** Each cell needs only the cell above (old value in the
   same slot) and the cell to the left (already updated).
5. **When does greedy fail for coin change?** When the coin system is not "canonical"; DP checks
   all last-coin choices.
