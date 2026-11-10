"""Queue, deque (ring buffer) and monotonic queue."""

from __future__ import annotations

from typing import Any


class Deque:
    """Double-ended queue on a circular array (ring buffer).

    `head` is the index of the first item; items occupy head, head+1, ...
    wrapping around with modulo. All four end operations are amortized O(1).
    """

    def __init__(self, capacity: int = 8) -> None:
        self._data: list[Any] = [None] * capacity
        self._head = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def _grow(self) -> None:
        """Double capacity, unrolling the ring so the first item sits at index 0."""
        old, n = self._data, len(self._data)
        self._data = [old[(self._head + i) % n] for i in range(self._size)] + [None] * n
        self._head = 0

    def append(self, x: Any) -> None:
        if self._size == len(self._data):
            self._grow()
        self._data[(self._head + self._size) % len(self._data)] = x
        self._size += 1

    def appendleft(self, x: Any) -> None:
        if self._size == len(self._data):
            self._grow()
        self._head = (self._head - 1) % len(self._data)
        self._data[self._head] = x
        self._size += 1

    def pop(self) -> Any:
        if not self._size:
            raise IndexError("pop from empty deque")
        i = (self._head + self._size - 1) % len(self._data)
        x, self._data[i] = self._data[i], None
        self._size -= 1
        return x

    def popleft(self) -> Any:
        if not self._size:
            raise IndexError("pop from empty deque")
        x, self._data[self._head] = self._data[self._head], None
        self._head = (self._head + 1) % len(self._data)
        self._size -= 1
        return x

    def __getitem__(self, i: int) -> Any:
        if i < 0:
            i += self._size
        if not 0 <= i < self._size:
            raise IndexError("deque index out of range")
        return self._data[(self._head + i) % len(self._data)]


class QueueWithTwoStacks:
    """FIFO queue from two LIFO stacks, amortized O(1) per operation.

    Push onto `inbox`. To pop, if `outbox` is empty pour all of `inbox` into
    it (which reverses the order), then pop from `outbox`. Every item is moved
    at most once, so the pouring cost averages out to O(1) per item.
    """

    def __init__(self) -> None:
        self._inbox: list[Any] = []
        self._outbox: list[Any] = []

    def push(self, x: Any) -> None:
        self._inbox.append(x)

    def _pour(self) -> None:
        if not self._outbox:
            while self._inbox:
                self._outbox.append(self._inbox.pop())

    def pop(self) -> Any:
        self._pour()
        if not self._outbox:
            raise IndexError("pop from empty queue")
        return self._outbox.pop()

    def peek(self) -> Any:
        self._pour()
        return self._outbox[-1]

    def __len__(self) -> int:
        return len(self._inbox) + len(self._outbox)


def sliding_window_max(nums: list[int], k: int) -> list[int]:
    """Maximum of every length-k window. O(n) with a monotonic deque.

    The deque stores indices whose values are DECREASING, so the front is
    always the window's maximum. A new value evicts every smaller value from
    the back: those can never be a maximum again because the new value is
    both larger and stays in the window longer.
    """
    dq = Deque()
    out: list[int] = []
    for i, x in enumerate(nums):
        while len(dq) and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:  # front index slid out of the window
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


def shortest_subarray_sum_at_least(nums: list[int], k: int) -> int:
    """Shortest subarray with sum >= k when nums may be NEGATIVE (-1 if none). O(n).

    Work on prefix sums P. We want the closest i < j with P[j] - P[i] >= k.
    Keep candidate i's in a deque with increasing P[i]:
    - if P[j] <= P[back], back is useless (j is later AND at least as small);
    - once the front satisfies P[j] - P[front] >= k, no later j can give a
      shorter answer with that front, so pop it.
    """
    prefix = [0]
    for x in nums:
        prefix.append(prefix[-1] + x)
    dq = Deque()
    best = len(nums) + 1
    for j, pj in enumerate(prefix):
        while len(dq) and pj - prefix[dq[0]] >= k:
            best = min(best, j - dq.popleft())
        while len(dq) and pj <= prefix[dq[-1]]:
            dq.pop()
        dq.append(j)
    return -1 if best > len(nums) else best
