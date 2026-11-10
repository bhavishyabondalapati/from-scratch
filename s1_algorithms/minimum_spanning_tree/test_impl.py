import random
from itertools import combinations

import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from .impl import kruskal, prim


def random_edges(rnd, n, m):
    edges = {}
    for _ in range(m):
        a, b = rnd.randrange(n), rnd.randrange(n)
        if a != b:
            edges[(min(a, b), max(a, b))] = rnd.randint(1, 20)
    return [(a, b, w) for (a, b), w in edges.items()]


def is_spanning_forest(n, edges, chosen):
    """chosen must be acyclic, a subset of edges, and connect whatever edges connects."""
    assert set(chosen) <= set(edges)
    label = list(range(n))
    for u, v, _ in chosen:
        assert label[u] != label[v], "cycle in the chosen edges"
        old, new = label[u], label[v]
        label = [new if x == old else x for x in label]
    full = csr_matrix(
        ([1] * len(edges), ([e[0] for e in edges], [e[1] for e in edges])), shape=(n, n)
    )
    n_components, _ = connected_components(full, directed=False)
    assert len(chosen) == n - n_components
    return True


@pytest.mark.parametrize("algorithm", [kruskal, prim])
def test_mst_weight_matches_scipy(algorithm):
    rnd = random.Random(0)
    for _ in range(200):
        n = rnd.randint(1, 12)
        edges = random_edges(rnd, n, rnd.randint(0, 30))
        total, chosen = algorithm(n, edges)
        matrix = csr_matrix(
            ([e[2] for e in edges], ([e[0] for e in edges], [e[1] for e in edges])),
            shape=(n, n),
            dtype=float,
        )
        assert total == pytest.approx(minimum_spanning_tree(matrix).sum())
        assert total == sum(w for _, _, w in chosen)
        normalised = [(min(u, v), max(u, v), w) for u, v, w in chosen]
        assert is_spanning_forest(n, edges, normalised)


def test_mst_against_brute_force_on_tiny_graphs():
    rnd = random.Random(1)
    for _ in range(60):
        n = rnd.randint(2, 6)
        edges = random_edges(rnd, n, 12)
        best = None
        for subset in combinations(edges, n - 1):  # try every set of n-1 edges
            label = list(range(n))
            ok = True
            for u, v, _ in subset:
                if label[u] == label[v]:
                    ok = False
                    break
                old, new = label[u], label[v]
                label = [new if x == old else x for x in label]
            if ok:
                weight = sum(w for _, _, w in subset)
                best = weight if best is None else min(best, weight)
        if best is not None:  # graph is connected
            assert kruskal(n, edges)[0] == prim(n, edges)[0] == best


def test_known_example():
    edges = [(0, 1, 4), (0, 2, 1), (1, 2, 2), (1, 3, 5), (2, 3, 8)]
    assert kruskal(4, edges)[0] == prim(4, edges)[0] == 8  # edges 1 + 2 + 5
