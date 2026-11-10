"""Trie (prefix tree)."""

from __future__ import annotations


class TrieNode:
    __slots__ = ("children", "is_word", "pass_count")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_word = False  # does a stored word end exactly here?
        self.pass_count = 0  # how many stored words pass through this node


class Trie:
    """A tree where each edge is a character and each root-to-node path is a prefix.

    Words that share a prefix share the same path, so prefix queries cost
    O(length of the prefix), no matter how many words are stored.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def _walk(self, prefix: str) -> TrieNode | None:
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def insert(self, word: str) -> bool:
        """Insert word; returns False if it was already present."""
        if self.search(word):
            return False
        node = self.root
        node.pass_count += 1
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
            node.pass_count += 1
        node.is_word = True
        return True

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_word

    def starts_with(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

    def count_prefix(self, prefix: str) -> int:
        """Number of stored words that start with prefix."""
        node = self._walk(prefix)
        return node.pass_count if node else 0

    def delete(self, word: str) -> bool:
        """Remove word; prune nodes no other word uses. Returns False if absent."""
        if not self.search(word):
            return False
        node = self.root
        node.pass_count -= 1
        for ch in word:
            child = node.children[ch]
            child.pass_count -= 1
            if child.pass_count == 0:
                del node.children[ch]  # nothing else lives below: cut the branch
                return True
            node = child
        node.is_word = False
        return True

    def words_with_prefix(self, prefix: str) -> list[str]:
        """All stored words starting with prefix, in sorted order (autocomplete)."""
        out: list[str] = []
        start = self._walk(prefix)
        if start is None:
            return out

        def dfs(node: TrieNode, path: list[str]) -> None:
            if node.is_word:
                out.append(prefix + "".join(path))
            for ch in sorted(node.children):
                path.append(ch)
                dfs(node.children[ch], path)
                path.pop()

        dfs(start, [])
        return out

    def search_with_wildcard(self, pattern: str) -> bool:
        """Like search, but '.' in the pattern matches any single character."""

        def dfs(node: TrieNode, i: int) -> bool:
            if i == len(pattern):
                return node.is_word
            ch = pattern[i]
            if ch == ".":
                return any(dfs(child, i + 1) for child in node.children.values())
            child = node.children.get(ch)
            return child is not None and dfs(child, i + 1)

        return dfs(self.root, 0)
