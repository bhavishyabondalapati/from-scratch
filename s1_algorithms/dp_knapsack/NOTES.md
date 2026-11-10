# Knapsack DP (0/1 and unbounded)

## Intuition
You have a bag with a weight limit and items with weights and values. Which items maximise the
value? Trying every subset is 2ⁿ. The DP insight: when deciding about one item, all that matters
about the earlier decisions is **how much capacity is left**, not which items used it up.

- **0/1:** each item at most once.
- **Unbounded:** unlimited copies of each item.

A surprising number of problems are knapsack in disguise: subset sum, equal partition, coin
change (min coins / number of ways), target sum, rod cutting.

## Core math
- 0/1: `best[i][c] = max(best[i-1][c], best[i-1][c - wᵢ] + vᵢ)`
- Unbounded: `best[i][c] = max(best[i-1][c], best[i][c - wᵢ] + vᵢ)` (note `i`, not `i-1`)
- In the 1D array the *only* difference is loop direction:
  - capacity **descending** → reads old values → each item once (0/1);
  - capacity **ascending** → reads values already updated with this item → reuse (unbounded).
- Counting variants replace `max` with `+`; feasibility variants replace it with `or`.
- Combinations vs permutations: **items outer** loop counts combinations; **capacity outer**
  loop counts ordered sequences.

## Complexity
O(n · W) time, O(W) space (O(n · W) if you need to reconstruct the chosen items).
This is **pseudo-polynomial**: polynomial in the numeric value W, exponential in the number of
bits of W. Knapsack is NP-hard in general.

## Common pitfalls
- Wrong loop direction in the 1D version (silently turns 0/1 into unbounded or vice versa).
- Greedy by value/weight ratio is optimal only for the *fractional* knapsack.
- Swapping the loop order when counting ways changes the meaning of the answer.
- Huge capacity: DP by weight is infeasible; DP over value or meet-in-the-middle may work.
- Zero-weight items / zeros in target-sum need care (each zero doubles the number of ways).

## When to use vs. alternatives
- "Pick a subset subject to a budget" with moderate integer budget → knapsack DP.
- Fractions allowed → greedy by ratio.
- n ≤ ~40 and huge weights → meet-in-the-middle (2^(n/2)).
- Many constraints → integer linear programming / branch and bound.

## Interview questions
1. **Why iterate capacity backwards for 0/1?** So `best[c - w]` has not yet been updated with the
   current item; iterating forward would let the item be taken repeatedly.
2. **Is O(nW) polynomial?** Pseudo-polynomial: W is exponential in its bit length.
3. **How do you recover which items were chosen?** Keep the 2D table and walk backwards: if
   `best[i][c] != best[i-1][c]`, item i was taken; subtract its weight.
4. **How do combinations vs permutations differ in coin change?** Coins-outer fixes an order of
   coin types (combinations); amount-outer allows any coin last (permutations).
5. **Why does greedy fail for 0/1 knapsack?** A high-ratio item can waste capacity; e.g.
   capacity 10, items (6, $7) and two (5, $5): greedy takes $7, optimal is $10.
