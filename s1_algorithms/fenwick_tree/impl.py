"""Fenwick tree (binary indexed tree): prefix sums with point updates in O(log n)."""

from __future__ import annotations


class FenwickTree:
    """Internally 1-indexed. tree[i] stores the sum of the `lowbit(i)` elements
    ending at position i, where lowbit(i) = i & -i is the lowest set bit of i.

    Example with i = 12 (binary 1100): lowbit = 4, so tree[12] covers 9..12.
    - prefix_sum(i): add tree[i], then strip the lowest bit (12 -> 8 -> 0).
    - add(i): update tree[i], then move to the next block containing i
      by adding the lowest bit (5 -> 6 -> 8 -> 16 ...).
    """

    def __init__(self, values: list[int] | int) -> None:
        if isinstance(values, int):
            values = [0] * values
        self.n = len(values)
        self.tree = [0, *values]
        # O(n) build: push each node's total into its immediate parent block.
        for i in range(1, self.n + 1):
            parent = i + (i & -i)
            if parent <= self.n:
                self.tree[parent] += self.tree[i]

    def add(self, index: int, delta: int) -> None:
        """values[index] += delta (0-indexed)."""
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, count: int) -> int:
        """Sum of the first `count` values, i.e. values[0:count]."""
        total, i = 0, count
        while i > 0:
            total += self.tree[i]
            i -= i & -i
        return total

    def range_sum(self, lo: int, hi: int) -> int:
        """Sum of values[lo:hi] (hi exclusive)."""
        return self.prefix_sum(hi) - self.prefix_sum(lo)

    def find_kth(self, k: int) -> int:
        """Smallest index i with prefix_sum(i + 1) >= k, for non-negative values.

        Binary lifting: decide the answer's bits from high to low, jumping over
        a whole block whenever its total still leaves us short of k. O(log n).
        Returns n if the total is less than k.
        """
        pos = 0
        step = 1 << self.n.bit_length()
        while step:
            nxt = pos + step
            if nxt <= self.n and self.tree[nxt] < k:
                pos = nxt
                k -= self.tree[nxt]
            step >>= 1
        return pos  # 1-indexed position pos + 1 is the answer -> 0-indexed pos


class RangeUpdateFenwick:
    """Range add + point query, by storing the DIFFERENCE array in a Fenwick tree.

    Adding v to values[lo:hi] is two point updates on the differences (+v at
    lo, -v at hi); the value at i is then the prefix sum of differences.
    """

    def __init__(self, n: int) -> None:
        self._diff = FenwickTree(n + 1)

    def range_add(self, lo: int, hi: int, delta: int) -> None:
        self._diff.add(lo, delta)
        self._diff.add(hi, -delta)

    def point_query(self, index: int) -> int:
        return self._diff.prefix_sum(index + 1)


def count_inversions(nums: list[int]) -> int:
    """Number of pairs i < j with nums[i] > nums[j]. O(n log n).

    Replace values by their rank, scan left to right, and for each element ask
    the Fenwick tree how many already-seen elements have a larger rank.
    """
    rank = {v: r for r, v in enumerate(sorted(set(nums)))}
    seen = FenwickTree(len(rank))
    inversions = 0
    for count_before, x in enumerate(nums):
        r = rank[x]
        inversions += count_before - seen.prefix_sum(r + 1)  # seen elements with rank > r
        seen.add(r, 1)
    return inversions
