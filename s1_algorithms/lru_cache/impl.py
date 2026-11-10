"""LRU cache: hash map + doubly linked list, O(1) get and put."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any


class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key: Hashable = None, value: Any = None) -> None:
        self.key, self.value = key, value
        self.prev: _Node | None = None
        self.next: _Node | None = None


class LRUCache:
    """Least-Recently-Used cache.

    - dict: key -> node, so we can find any entry in O(1).
    - doubly linked list ordered by recency: most recent right after `head`,
      least recent right before `tail`. A doubly linked list lets us unlink a
      node from the middle in O(1) because the node knows both neighbours.

    `head` and `tail` are sentinel (dummy) nodes so we never handle None edges.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._map: dict[Hashable, _Node] = {}
        self._head, self._tail = _Node(), _Node()
        self._head.next, self._tail.prev = self._tail, self._head

    def _unlink(self, node: _Node) -> None:
        node.prev.next, node.next.prev = node.next, node.prev

    def _push_front(self, node: _Node) -> None:
        node.prev, node.next = self._head, self._head.next
        self._head.next.prev = node
        self._head.next = node

    def get(self, key: Hashable, default: Any = None) -> Any:
        node = self._map.get(key)
        if node is None:
            return default
        self._unlink(node)  # reading counts as "use": move to the front
        self._push_front(node)
        return node.value

    def put(self, key: Hashable, value: Any) -> None:
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self._map) == self.capacity:
            lru = self._tail.prev  # least recently used lives at the back
            self._unlink(lru)
            del self._map[lru.key]  # this is why nodes store their key
        node = _Node(key, value)
        self._map[key] = node
        self._push_front(node)

    def __len__(self) -> int:
        return len(self._map)

    def keys_by_recency(self) -> list[Hashable]:
        """Keys from most to least recently used (handy for tests/debugging)."""
        out, node = [], self._head.next
        while node is not self._tail:
            out.append(node.key)
            node = node.next
        return out
