# Pattern: backtracking

**Signal:** "all possible", "every combination/arrangement", "generate", constraint puzzles,
small input sizes (n ≤ ~20).

## Representative LeetCode problems (names only)
- Subsets
- Permutations
- Combination Sum
- N-Queens
- Sudoku Solver

## Solved example (my own wording)
**Problem.** A café sells items with fixed prices. A customer has a voucher worth exactly `v` and
wants to see every different basket (items may repeat) whose total is exactly `v`.

**Thinking.** Build a basket one item at a time. To avoid listing the same basket in different
orders, only allow items at or after the position of the last item added. Sort prices so that as
soon as one price is too big for what is left of the voucher, I can stop looking further.

```python
def baskets(prices, v):
    prices = sorted(set(prices))
    found, basket = [], []

    def explore(start, left):
        if left == 0:
            found.append(basket.copy())
            return
        for i in range(start, len(prices)):
            if prices[i] > left:
                break
            basket.append(prices[i])
            explore(i, left - prices[i])
            basket.pop()

    explore(0, v)
    return found


baskets([2, 3, 5], 8)  # [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
```

**Complexity.** Exponential in the worst case (it lists every answer), with pruning from sorting.
