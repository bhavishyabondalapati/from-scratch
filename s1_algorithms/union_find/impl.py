"""Union-find (disjoint set union) with union by rank and path compression."""

from __future__ import annotations


class UnionFind:
    """Tracks a partition of {0..n-1} into groups that can only merge.

    Each group is a tree stored via parent pointers; the root is the group's
    representative. Two optimisations make it almost O(1) per operation:
    - path compression: after find(x), point everything on the path at the root;
    - union by rank: attach the shallower tree under the deeper one.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))  # everyone starts as their own root
        self.rank = [0] * n  # upper bound on the height of each root's tree
        self.size = [1] * n  # number of elements in each root's group
        self.components = n

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # second pass: compress the path
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        """Merge the groups of a and b. Returns False if already together."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra  # make ra the deeper tree
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1  # equal heights: the merged tree is one taller
        self.components -= 1
        return True

    def connected(self, a: int, b: int) -> bool:
        return self.find(a) == self.find(b)

    def group_size(self, x: int) -> int:
        return self.size[self.find(x)]


def count_components(n: int, edges: list[tuple[int, int]]) -> int:
    """Number of connected components in an undirected graph."""
    uf = UnionFind(n)
    for a, b in edges:
        uf.union(a, b)
    return uf.components


def find_redundant_edge(n: int, edges: list[tuple[int, int]]) -> tuple[int, int] | None:
    """First edge that closes a cycle (both ends already connected), else None."""
    uf = UnionFind(n)
    for a, b in edges:
        if not uf.union(a, b):
            return (a, b)
    return None


def merge_accounts(accounts: list[list[str]]) -> list[list[str]]:
    """Each account is [name, email1, email2, ...]. Accounts sharing any email
    belong to the same person; return merged [name, sorted emails...] lists."""
    uf = UnionFind(len(accounts))
    owner: dict[str, int] = {}  # email -> first account index that listed it
    for i, (_, *emails) in enumerate(accounts):
        for email in emails:
            if email in owner:
                uf.union(i, owner[email])
            else:
                owner[email] = i
    groups: dict[int, list[str]] = {}
    for email, i in owner.items():
        groups.setdefault(uf.find(i), []).append(email)
    return [[accounts[root][0], *sorted(emails)] for root, emails in groups.items()]
