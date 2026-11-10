"""Arrays & hashing: a hash map from scratch plus the classic "use a dict/set" problems."""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from typing import Any


class HashMap:
    """A hash map using separate chaining (each bucket is a small list of pairs).

    Average O(1) get/put/delete. When the load factor (items / buckets) passes
    0.75 we double the bucket array and re-insert everything ("rehash").
    """

    def __init__(self, capacity: int = 8) -> None:
        self._buckets: list[list[tuple[Hashable, Any]]] = [[] for _ in range(capacity)]
        self._size = 0

    def _bucket(self, key: Hashable) -> list[tuple[Hashable, Any]]:
        return self._buckets[hash(key) % len(self._buckets)]

    def put(self, key: Hashable, value: Any) -> None:
        bucket = self._bucket(key)
        for i, (k, _) in enumerate(bucket):
            if k == key:  # key already present: overwrite
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1
        if self._size > 0.75 * len(self._buckets):
            self._resize()

    def get(self, key: Hashable, default: Any = None) -> Any:
        for k, v in self._bucket(key):
            if k == key:
                return v
        return default

    def delete(self, key: Hashable) -> bool:
        """Remove key. Returns True if it was present."""
        bucket = self._bucket(key)
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self._size -= 1
                return True
        return False

    def _resize(self) -> None:
        old = self._buckets
        self._buckets = [[] for _ in range(2 * len(old))]
        for bucket in old:
            for k, v in bucket:
                self._bucket(k).append((k, v))

    def __contains__(self, key: Hashable) -> bool:
        return any(k == key for k, _ in self._bucket(key))

    def __len__(self) -> int:
        return self._size

    def items(self) -> Iterator[tuple[Hashable, Any]]:
        for bucket in self._buckets:
            yield from bucket


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    """Indices (i, j), i < j, with nums[i] + nums[j] == target. O(n) time."""
    seen: dict[int, int] = {}  # value -> index where we saw it
    for j, x in enumerate(nums):
        if target - x in seen:
            return seen[target - x], j
        seen[x] = j
    return None


def group_anagrams(words: list[str]) -> list[list[str]]:
    """Group words that are anagrams. Key = letter counts, so O(total letters)."""
    groups: dict[tuple[int, ...], list[str]] = {}
    for w in words:
        counts = [0] * 26
        for ch in w:
            counts[ord(ch) - ord("a")] += 1
        groups.setdefault(tuple(counts), []).append(w)
    return list(groups.values())


def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """The k most frequent values using bucket sort: O(n), no heap needed.

    buckets[f] holds every value that appears exactly f times; a value can
    appear at most n times, so there are only n + 1 buckets to scan.
    """
    freq: dict[int, int] = {}
    for x in nums:
        freq[x] = freq.get(x, 0) + 1
    buckets: list[list[int]] = [[] for _ in range(len(nums) + 1)]
    for value, f in freq.items():
        buckets[f].append(value)
    out: list[int] = []
    for f in range(len(nums), 0, -1):
        for value in buckets[f]:
            out.append(value)
            if len(out) == k:
                return out
    return out


def longest_consecutive(nums: list[int]) -> int:
    """Length of the longest run x, x+1, x+2, ... present in nums. O(n).

    Only start counting from x when x-1 is absent, so each run is walked once.
    """
    present = set(nums)
    best = 0
    for x in present:
        if x - 1 not in present:
            length = 1
            while x + length in present:
                length += 1
            best = max(best, length)
    return best


def product_except_self(nums: list[int]) -> list[int]:
    """out[i] = product of all nums except nums[i], without division. O(n)."""
    n = len(nums)
    out = [1] * n
    prefix = 1
    for i in range(n):  # out[i] = product of everything to the left
        out[i] = prefix
        prefix *= nums[i]
    suffix = 1
    for i in range(n - 1, -1, -1):  # multiply in everything to the right
        out[i] *= suffix
        suffix *= nums[i]
    return out
