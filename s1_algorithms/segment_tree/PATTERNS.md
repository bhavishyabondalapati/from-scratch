# Pattern: segment tree

**Signal:** many interleaved "update" and "range query" operations; range min/max; range updates.

## Representative LeetCode problems (names only)
- Range Sum Query - Mutable
- Range Sum Query 2D - Mutable
- Count of Smaller Numbers After Self
- My Calendar III
- Falling Squares

## Solved example (my own wording)
**Problem.** A row of `n` sensors each reports a temperature. Readings change over time. Support
`set(i, t)` and `coldest(lo, hi)` = the minimum reading among sensors `lo .. hi-1`.

**Thinking.** Minimum cannot be "subtracted", so prefix tricks fail. Store sensors in the leaves
of a tree kept in an array of size `2n`; each parent stores the min of its two children. A change
walks up from a leaf fixing parents. A query climbs from both ends of the range, taking a node
whenever its parent would stick out of the range.

```python
class ColdestTracker:
    def __init__(self, temps):
        self.n = len(temps)
        self.t = [float("inf")] * self.n + list(temps)
        for i in range(self.n - 1, 0, -1):
            self.t[i] = min(self.t[2 * i], self.t[2 * i + 1])

    def set(self, i, temp):
        i += self.n
        self.t[i] = temp
        while i > 1:
            i //= 2
            self.t[i] = min(self.t[2 * i], self.t[2 * i + 1])

    def coldest(self, lo, hi):
        best = float("inf")
        lo, hi = lo + self.n, hi + self.n
        while lo < hi:
            if lo & 1:
                best = min(best, self.t[lo])
                lo += 1
            if hi & 1:
                hi -= 1
                best = min(best, self.t[hi])
            lo, hi = lo // 2, hi // 2
        return best


c = ColdestTracker([5, 3, 8, 6])
c.coldest(0, 4)  # 3
c.set(1, 9)
c.coldest(0, 4)  # 5
```

**Complexity.** O(log n) per operation, O(n) memory.
