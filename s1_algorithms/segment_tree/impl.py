"""Segment tree: range queries with point updates, plus a lazy version for range updates."""

from __future__ import annotations

from collections.abc import Callable


class SegmentTree:
    """Iterative segment tree for any associative operation (sum, min, max, gcd...).

    Layout: an array of size 2n. Leaves live at tree[n : 2n]; internal node i
    combines its children 2i and 2i + 1. `identity` is the value that changes
    nothing when combined (0 for sum, +inf for min, -inf for max).
    """

    def __init__(
        self,
        values: list[int],
        op: Callable[[int, int], int] = lambda a, b: a + b,
        identity: int = 0,
    ) -> None:
        self.n = len(values)
        self.op, self.identity = op, identity
        self.tree = [identity] * self.n + list(values)
        for i in range(self.n - 1, 0, -1):  # build parents from the leaves upward
            self.tree[i] = op(self.tree[2 * i], self.tree[2 * i + 1])

    def update(self, index: int, value: int) -> None:
        """Set values[index] = value and repair every ancestor. O(log n)."""
        i = index + self.n
        self.tree[i] = value
        i //= 2
        while i >= 1:
            self.tree[i] = self.op(self.tree[2 * i], self.tree[2 * i + 1])
            i //= 2

    def query(self, lo: int, hi: int) -> int:
        """Combine values[lo:hi] (hi exclusive). O(log n).

        Walk both ends of the range up the tree. If `lo` is a right child, its
        parent also covers things left of the range, so take lo itself and step
        right. Symmetric for `hi`. Left and right partial results are kept
        separate so the order is right for non-commutative operations.
        """
        left, right = self.identity, self.identity
        lo += self.n
        hi += self.n
        while lo < hi:
            if lo & 1:
                left = self.op(left, self.tree[lo])
                lo += 1
            if hi & 1:
                hi -= 1
                right = self.op(self.tree[hi], right)
            lo //= 2
            hi //= 2
        return self.op(left, right)


class LazySegmentTree:
    """Range add + range sum, both O(log n), using lazy propagation.

    A range update stops at nodes fully inside the range, records "everything
    below me still owes +v" in lazy[node], and only pushes that debt down to
    the children when a later operation needs to look inside the node.
    """

    def __init__(self, values: list[int]) -> None:
        self.n = len(values)
        self.total = [0] * (4 * max(1, self.n))  # sum of each node's segment
        self.lazy = [0] * (4 * max(1, self.n))  # pending add for each node's children
        if self.n:
            self._build(1, 0, self.n - 1, values)

    def _build(self, node: int, lo: int, hi: int, values: list[int]) -> None:
        if lo == hi:
            self.total[node] = values[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, values)
        self._build(2 * node + 1, mid + 1, hi, values)
        self.total[node] = self.total[2 * node] + self.total[2 * node + 1]

    def _push_down(self, node: int, lo: int, hi: int) -> None:
        if self.lazy[node]:
            mid = (lo + hi) // 2
            for child, size in ((2 * node, mid - lo + 1), (2 * node + 1, hi - mid)):
                self.total[child] += self.lazy[node] * size
                self.lazy[child] += self.lazy[node]
            self.lazy[node] = 0

    def range_add(self, left: int, right: int, value: int) -> None:
        """Add value to every element of values[left:right] (right exclusive)."""
        if left < right:
            self._add(1, 0, self.n - 1, left, right - 1, value)

    def _add(self, node: int, lo: int, hi: int, left: int, right: int, value: int) -> None:
        if right < lo or hi < left:  # no overlap
            return
        if left <= lo and hi <= right:  # node fully covered: update and stop here
            self.total[node] += value * (hi - lo + 1)
            self.lazy[node] += value
            return
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        self._add(2 * node, lo, mid, left, right, value)
        self._add(2 * node + 1, mid + 1, hi, left, right, value)
        self.total[node] = self.total[2 * node] + self.total[2 * node + 1]

    def range_sum(self, left: int, right: int) -> int:
        """Sum of values[left:right] (right exclusive)."""
        if left >= right:
            return 0
        return self._sum(1, 0, self.n - 1, left, right - 1)

    def _sum(self, node: int, lo: int, hi: int, left: int, right: int) -> int:
        if right < lo or hi < left:
            return 0
        if left <= lo and hi <= right:
            return self.total[node]
        self._push_down(node, lo, hi)
        mid = (lo + hi) // 2
        return self._sum(2 * node, lo, mid, left, right) + self._sum(
            2 * node + 1, mid + 1, hi, left, right
        )
