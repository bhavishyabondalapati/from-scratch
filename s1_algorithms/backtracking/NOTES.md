# Backtracking

## Intuition
Backtracking is organised trial and error. Picture a tree of decisions: at each level you make one
choice (take this number or not, put the queen in this column, write this digit). You walk down
the tree; if you reach a dead end you step back one level, undo the last choice, and try the next
option. The power comes from **pruning**: detecting a dead end early cuts off the whole subtree
below it.

## Core idea — the template
```
def explore(state):
    if state is a complete answer: record a COPY; return
    for choice in choices(state):
        if not valid(choice): continue      # prune
        apply(choice)                       # choose
        explore(state)                      # explore
        undo(choice)                        # un-choose
```
- Subsets: binary take/skip decisions → 2ⁿ leaves.
- Permutations: pick any unused element → n! leaves.
- Combinations: pick in increasing order (a `start` index) so each set is generated once.
- Duplicates: sort, then skip a value equal to the previous one *at the same depth*.
- N-Queens: one queen per row; columns and both diagonals tracked in sets (`r - c`, `r + c`).
- Sudoku: choose the most constrained cell first (fewest candidates).

## Complexity
| Problem | Time | Space (besides output) |
|---|---|---|
| subsets | O(n · 2ⁿ) | O(n) |
| permutations | O(n · n!) | O(n) |
| combinations | O(k · C(n, k)) | O(k) |
| N-Queens | O(n!) worst case | O(n) |
| Sudoku | exponential worst case; fast in practice with pruning | O(81) |

## Common pitfalls
- Appending `current` instead of `current.copy()` → every answer ends up identical (and empty).
- Forgetting to undo a choice (state leaks into sibling branches).
- Generating duplicates: missing the `start` index (combinations) or the same-depth skip rule.
- Checking validity only at the leaves instead of pruning as early as possible.

## When to use vs. alternatives
- Need to **enumerate all** solutions, or constraints are irregular → backtracking.
- Only need a count/optimum and subproblems overlap → dynamic programming (often bitmask DP).
- Greedy works if a local choice is provably safe.
- Very hard constraint problems → constraint propagation / SAT solvers (same idea, more pruning).

## Interview questions
1. **Backtracking vs brute force?** Same search space, but backtracking abandons partial
   candidates as soon as they violate a constraint.
2. **How do you avoid duplicate permutations/subsets with repeated values?** Sort and skip
   `nums[i] == nums[i-1]` when the previous one was not chosen at this depth.
3. **Why are diagonals `r - c` and `r + c`?** Moving along "\" increases r and c together
   (difference constant); along "/" one rises as the other falls (sum constant).
4. **How do you speed up Sudoku?** Most-constrained-cell ordering, O(1) validity via row/col/box
   sets (or bitmasks), and constraint propagation.
5. **Time complexity of generating all subsets, and can it be better?** O(n · 2ⁿ); no, because
   the output itself is that large.
