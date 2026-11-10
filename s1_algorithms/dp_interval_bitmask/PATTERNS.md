# Pattern: interval DP and bitmask DP

**Signal (interval):** "merge adjacent", "cut the stick/string", "burst/remove and score by
neighbours", best way to parenthesise.
**Signal (bitmask):** "visit all", "assign each exactly once", n ≤ 20, state needs a set.

## Representative LeetCode problems (names only)
- Burst Balloons
- Minimum Cost to Cut a Stick
- Palindrome Partitioning II
- Partition to K Equal Sum Subsets
- Shortest Path Visiting All Nodes

## Solved example (my own wording)
**Problem.** A carpenter must cut a plank of length `L` at several marked positions. Each cut
costs the length of the piece being cut at that moment. In what order should the cuts be made so
the total cost is smallest? (Return the cost.)

**Thinking.** Add the plank's ends to the list of marks and sort. Consider the piece between mark
`i` and mark `j`. Whatever cut is made *first* inside it costs the piece's whole length and
leaves two independent smaller pieces. Try every first cut; solve short pieces before long ones.

```python
def cheapest_cuts(L, marks):
    pts = [0, *sorted(marks), L]
    n = len(pts)
    cost = [[0] * n for _ in range(n)]
    for gap in range(2, n):
        for i in range(n - gap):
            j = i + gap
            cost[i][j] = pts[j] - pts[i] + min(cost[i][k] + cost[k][j] for k in range(i + 1, j))
    return cost[0][n - 1]


cheapest_cuts(10, [2, 4, 7])  # 20: cut at 4 (10), then 2 (4), then 7 (6)
```

**Complexity.** O(m³) for m marks.
