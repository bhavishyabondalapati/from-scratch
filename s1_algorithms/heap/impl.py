"""Binary heap from scratch, plus top-k, merge k sorted lists and median of a stream."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


class Heap:
    """Binary min-heap stored in a flat list.

    The list represents a complete binary tree level by level:
        parent(i) = (i - 1) // 2,  children(i) = 2i + 1 and 2i + 2
    Heap property: every parent's key <= its children's keys, so the minimum
    is always at index 0. Pass key=lambda x: -x for a max-heap.
    """

    def __init__(self, items: Iterable[Any] = (), key: Callable[[Any], Any] = lambda x: x) -> None:
        self._key = key
        self._a: list[Any] = list(items)
        # Heapify bottom-up: sift down every internal node, last parent first.
        # This is O(n), not O(n log n), because most nodes are near the bottom
        # and have only a short distance to fall.
        for i in range(len(self._a) // 2 - 1, -1, -1):
            self._sift_down(i)

    def __len__(self) -> int:
        return len(self._a)

    def peek(self) -> Any:
        if not self._a:
            raise IndexError("peek at empty heap")
        return self._a[0]

    def push(self, x: Any) -> None:
        self._a.append(x)  # put it at the bottom...
        self._sift_up(len(self._a) - 1)  # ...and let it float up

    def pop(self) -> Any:
        if not self._a:
            raise IndexError("pop from empty heap")
        top, last = self._a[0], self._a.pop()
        if self._a:
            self._a[0] = last  # move the last leaf to the root...
            self._sift_down(0)  # ...and let it sink
        return top

    def _sift_up(self, i: int) -> None:
        a, key = self._a, self._key
        while i > 0:
            parent = (i - 1) // 2
            if key(a[i]) >= key(a[parent]):
                break
            a[i], a[parent] = a[parent], a[i]
            i = parent

    def _sift_down(self, i: int) -> None:
        a, key, n = self._a, self._key, len(self._a)
        while True:
            smallest = i
            for child in (2 * i + 1, 2 * i + 2):
                if child < n and key(a[child]) < key(a[smallest]):
                    smallest = child
            if smallest == i:
                return
            a[i], a[smallest] = a[smallest], a[i]
            i = smallest


def top_k_largest(nums: Iterable[int], k: int) -> list[int]:
    """The k largest values, largest first. O(n log k) time, O(k) space.

    Keep a MIN-heap of size k: its root is the weakest of the current top k,
    so a new value only needs to beat the root to get in.
    """
    if k <= 0:
        return []
    heap = Heap()
    for x in nums:
        if len(heap) < k:
            heap.push(x)
        elif x > heap.peek():
            heap.pop()
            heap.push(x)
    return sorted((heap.pop() for _ in range(len(heap))), reverse=True)


def merge_k_sorted(lists: list[list[int]]) -> list[int]:
    """Merge k sorted lists. O(N log k) where N is the total number of items.

    The heap holds one "front" item per list as (value, list index, item index).
    """
    heap = Heap((lst[0], i, 0) for i, lst in enumerate(lists) if lst)
    out: list[int] = []
    while len(heap):
        value, i, j = heap.pop()
        out.append(value)
        if j + 1 < len(lists[i]):
            heap.push((lists[i][j + 1], i, j + 1))
    return out


class MedianFinder:
    """Running median with two heaps.

    `low` is a max-heap holding the smaller half, `high` is a min-heap holding
    the larger half. Invariants: every item in low <= every item in high, and
    len(low) is equal to len(high) or exactly one more. Then the median is
    low's top (odd count) or the average of both tops (even count).
    """

    def __init__(self) -> None:
        self._low = Heap(key=lambda x: -x)
        self._high = Heap()

    def add(self, x: float) -> None:
        # Route x through low so the largest of the small half moves to high...
        self._low.push(x)
        self._high.push(self._low.pop())
        # ...then rebalance sizes so low never has fewer items than high.
        if len(self._high) > len(self._low):
            self._low.push(self._high.pop())

    def median(self) -> float:
        if not len(self._low):
            raise ValueError("no data")
        if len(self._low) > len(self._high):
            return float(self._low.peek())
        return (self._low.peek() + self._high.peek()) / 2
