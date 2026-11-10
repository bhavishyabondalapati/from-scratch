# Pattern: knapsack

**Signal:** choose a subset (or multiset) of items under a capacity/target; "can you make sum X",
"number of ways to make X", "fewest items to make X".

## Representative LeetCode problems (names only)
- Partition Equal Subset Sum
- Coin Change II
- Target Sum
- Ones and Zeroes
- Last Stone Weight II

## Solved example (my own wording)
**Problem.** Two siblings inherit a set of indivisible antiques with known appraisal values. Can
the antiques be divided so both receive exactly the same total value?

**Thinking.** If the total is odd, no. Otherwise one sibling must receive exactly half, so the
question is "is there a subset summing to `total / 2`?" Track which sums are reachable. Each
antique can be used once, so sweep the sums from high to low when adding it.

```python
def fair_split(values):
    total = sum(values)
    if total % 2:
        return False
    half = total // 2
    reachable = [True] + [False] * half
    for v in values:
        for s in range(half, v - 1, -1):
            reachable[s] = reachable[s] or reachable[s - v]
    return reachable[half]


fair_split([300, 500, 200, 400])  # True  (300 + 400 = 500 + 200)
fair_split([300, 500, 100])  # False
```

**Complexity.** O(n · total) time, O(total) space.
