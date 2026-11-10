# Pattern: 1D DP and grid DP

**Signal:** "number of ways", "minimum/maximum ... to reach", "can it be done", choices made
left-to-right or cell-by-cell where each choice depends on a few previous ones.

## Representative LeetCode problems (names only)
- Climbing Stairs
- House Robber
- Coin Change
- Unique Paths
- Maximal Square

## Solved example (my own wording)
**Problem.** A gardener has a row of flower beds, each with a known yield. Watering two
neighbouring beds on the same day floods the path between them, so no two adjacent beds may be
watered. What is the highest total yield?

**Thinking.** State: `best[i]` = highest yield using only the first `i` beds. For the last bed,
either leave it dry (`best[i-1]`) or water it, in which case the bed before it must be dry
(`best[i-2] + yield`). Only the last two values are needed.

```python
def best_yield(yields):
    two_back, one_back = 0, 0
    for y in yields:
        two_back, one_back = one_back, max(one_back, two_back + y)
    return one_back


best_yield([4, 9, 3, 1, 8])  # 17 (beds with 9 and 8)
```

**Complexity.** O(n) time, O(1) space.
