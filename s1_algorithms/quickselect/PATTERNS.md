# Pattern: quickselect / selection

**Signal:** "k-th largest/smallest", "median", "k closest", "top k" when the full sorted order is
not required and all data is in memory.

## Representative LeetCode problems (names only)
- Kth Largest Element in an Array
- K Closest Points to Origin
- Top K Frequent Elements
- Wiggle Sort II
- Minimum Moves to Equal Array Elements II

## Solved example (my own wording)
**Problem.** A delivery company wants to place one depot on a straight road so that the total
driving distance to all customers (given by their positions on the road) is as small as possible.

**Thinking.** The total distance Σ|x − pᵢ| is minimised at the **median** position: moving the
depot one step toward the side with more customers always reduces the total. So I only need the
median, which selection finds in O(n) on average without sorting everything.

```python
import random


def kth_smallest(a, k):
    a = list(a)
    while True:
        pivot = random.choice(a)
        smaller = [x for x in a if x < pivot]
        equal = [x for x in a if x == pivot]
        if k < len(smaller):
            a = smaller
        elif k < len(smaller) + len(equal):
            return pivot
        else:
            k -= len(smaller) + len(equal)
            a = [x for x in a if x > pivot]


def best_depot(positions):
    depot = kth_smallest(positions, len(positions) // 2)
    return depot, sum(abs(depot - p) for p in positions)


best_depot([2, 9, 4, 30, 5])  # (5, 33)
```

**Complexity.** O(n) average time.
