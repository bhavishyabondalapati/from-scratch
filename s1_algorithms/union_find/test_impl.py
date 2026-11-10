import random

from .impl import UnionFind, count_components, find_redundant_edge, merge_accounts


class NaiveGroups:
    """Reference: relabel a whole group on every union (O(n) per union)."""

    def __init__(self, n):
        self.label = list(range(n))

    def union(self, a, b):
        la, lb = self.label[a], self.label[b]
        if la == lb:
            return False
        self.label = [la if x == lb else x for x in self.label]
        return True


def test_union_find_matches_naive_reference():
    rnd = random.Random(0)
    for n in [1, 2, 10, 60]:
        uf, ref = UnionFind(n), NaiveGroups(n)
        for _ in range(4 * n):
            a, b = rnd.randrange(n), rnd.randrange(n)
            assert uf.union(a, b) == ref.union(a, b)
            x, y = rnd.randrange(n), rnd.randrange(n)
            assert uf.connected(x, y) == (ref.label[x] == ref.label[y])
            assert uf.group_size(x) == ref.label.count(ref.label[x])
            assert uf.components == len(set(ref.label))


def test_path_compression_flattens_and_rank_bounds_height():
    n = 1024
    uf = UnionFind(n)
    for i in range(n - 1):
        uf.union(i, i + 1)
    # Union by rank guarantees height <= log2(n) even before compression.
    assert max(uf.rank) <= 10
    root = uf.find(0)
    for i in range(n):
        uf.find(i)
    assert all(uf.parent[i] == root for i in range(n))  # fully flattened


def test_count_components_against_dfs():
    rnd = random.Random(1)
    for _ in range(200):
        n = rnd.randint(1, 15)
        edges = [(rnd.randrange(n), rnd.randrange(n)) for _ in range(rnd.randint(0, 15))]
        adj = {i: set() for i in range(n)}
        for a, b in edges:
            adj[a].add(b)
            adj[b].add(a)
        seen, comps = set(), 0
        for s in range(n):
            if s not in seen:
                comps += 1
                stack = [s]
                while stack:
                    u = stack.pop()
                    if u not in seen:
                        seen.add(u)
                        stack.extend(adj[u])
        assert count_components(n, edges) == comps


def test_find_redundant_edge():
    assert find_redundant_edge(3, [(0, 1), (0, 2), (1, 2)]) == (1, 2)
    assert find_redundant_edge(4, [(0, 1), (1, 2), (2, 3)]) is None
    assert find_redundant_edge(5, [(0, 1), (1, 2), (2, 3), (0, 3), (0, 4)]) == (0, 3)


def test_merge_accounts():
    accounts = [
        ["Ana", "a@x.com", "ana@work.com"],
        ["Ana", "ana@old.com"],  # a different Ana: no shared email
        ["Ana", "ana@work.com", "a2@x.com"],
        ["Bo", "bo@x.com"],
        ["Ana", "a2@x.com", "a3@x.com"],  # linked transitively
    ]
    merged = sorted(merge_accounts(accounts))
    assert merged == [
        ["Ana", "a2@x.com", "a3@x.com", "a@x.com", "ana@work.com"],
        ["Ana", "ana@old.com"],
        ["Bo", "bo@x.com"],
    ]
