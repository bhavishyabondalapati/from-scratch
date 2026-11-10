"""LFU cache with O(1) get and put."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Hashable
from typing import Any


class LFUCache:
    """Least-Frequently-Used cache; ties are broken by least-recently-used.

    Three pieces of state:
    - values[key] -> value
    - freq[key]   -> how many times key has been used
    - groups[f]   -> keys used exactly f times, in order of last use (oldest first)
    - min_freq    -> the smallest f with a non-empty group (who to evict)

    Each group is an OrderedDict used as an ordered set: O(1) add at the end,
    O(1) delete by key, O(1) pop of the oldest. (OrderedDict is itself a hash
    map + doubly linked list, exactly the structure built by hand in lru_cache.)

    The trick that keeps eviction O(1): min_freq only ever changes in two
    simple ways. A new key resets it to 1, and when the min group empties
    because its key was bumped, the new minimum is exactly min_freq + 1.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._values: dict[Hashable, Any] = {}
        self._freq: dict[Hashable, int] = {}
        self._groups: dict[int, OrderedDict[Hashable, None]] = {}
        self._min_freq = 0

    def _touch(self, key: Hashable) -> None:
        """Move key from its frequency group f to group f + 1."""
        f = self._freq[key]
        del self._groups[f][key]
        if not self._groups[f]:
            del self._groups[f]
            if self._min_freq == f:
                self._min_freq = f + 1
        self._freq[key] = f + 1
        self._groups.setdefault(f + 1, OrderedDict())[key] = None

    def get(self, key: Hashable, default: Any = None) -> Any:
        if key not in self._values:
            return default
        self._touch(key)
        return self._values[key]

    def put(self, key: Hashable, value: Any) -> None:
        if self.capacity <= 0:
            return
        if key in self._values:
            self._values[key] = value
            self._touch(key)
            return
        if len(self._values) == self.capacity:
            group = self._groups[self._min_freq]
            victim, _ = group.popitem(last=False)  # oldest key in the min group
            if not group:
                del self._groups[self._min_freq]
            del self._values[victim], self._freq[victim]
        self._values[key] = value
        self._freq[key] = 1
        self._groups.setdefault(1, OrderedDict())[key] = None
        self._min_freq = 1

    def __len__(self) -> int:
        return len(self._values)
