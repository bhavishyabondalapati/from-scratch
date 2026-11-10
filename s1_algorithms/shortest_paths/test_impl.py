import random

import numpy as np
import pytest
from scipy.sparse import csgraph, csr_matrix

from .impl import INF, bellman_ford, dijkstra, floyd_warshall, reconstruct_path


def random_weighted_graph(rnd, n, m, low, high):
    """Returns (adjacency list, scipy CSR matrix) for the same directed graph."""
    edges = {}
    for _ in range(m):
        a, b = rnd.randrange(n), rnd.randrange(n)
        if a != b:
            w = rnd.randint(low, high)
            edges[(a, b)] = w if w != 0 else 1  # scipy treats explicit zeros awkwardly
    graph = [[] for _ in range(n)]
    for (a, b), w in edges.items():
        graph[a].append((b, w))
    rows = [a for a, _ in edges]
    cols = [b for _, b in edges]
    matrix = csr_matrix((list(edges.values()), (rows, cols)), shape=(n, n), dtype=float)
    return graph, matrix


def test_dijkstra_matches_scipy_and_paths_are_consistent():
    rnd = random.Random(0)
    for _ in range(150):
        n = rnd.randint(1, 12)
        graph, matrix = random_weighted_graph(rnd, n, rnd.randint(0, 30), 1, 20)
        source = rnd.randrange(n)
        dist, parent = dijkstra(graph, source)
        np.testing.assert_allclose(dist, csgraph.dijkstra(matrix, indices=source))
        weight = {(u, v): w for u in range(n) for v, w in graph[u]}
        for target in range(n):
            path = reconstruct_path(parent, source, target)
            if dist[target] == INF:
                assert path is None
            else:
                assert path[0] == source and path[-1] == target
                assert sum(weight[e] for e in zip(path, path[1:])) == dist[target]


def test_dijkstra_rejects_negative_weights():
    with pytest.raises(ValueError):
        dijkstra([[(1, -1)], []], 0)


def test_bellman_ford_and_floyd_warshall_match_scipy_with_negative_edges():
    rnd = random.Random(1)
    negative_cycles = 0
    for _ in range(200):
        n = rnd.randint(1, 9)
        graph, matrix = random_weighted_graph(rnd, n, rnd.randint(0, 16), -3, 12)
        try:
            ref_all = csgraph.floyd_warshall(matrix)
            has_negative_cycle = bool((np.diag(ref_all) < 0).any())
        except csgraph.NegativeCycleError:
            has_negative_cycle = True
        ours_all = floyd_warshall(graph)
        if has_negative_cycle:
            negative_cycles += 1
            assert ours_all is None
            continue
        np.testing.assert_allclose(ours_all, ref_all)
        for source in range(n):
            np.testing.assert_allclose(bellman_ford(graph, source), ref_all[source])
    assert negative_cycles > 5  # the random graphs did exercise that branch


def test_bellman_ford_negative_cycle_only_matters_if_reachable():
    #  0 -> 1 (weight 1);   2 <-> 3 form a negative cycle that 0 cannot reach.
    graph = [[(1, 1)], [], [(3, -2)], [(2, 1)]]
    assert bellman_ford(graph, 0) == [0, 1, INF, INF]
    assert bellman_ford(graph, 2) is None
    assert floyd_warshall(graph) is None


def test_all_three_agree_on_non_negative_graphs():
    rnd = random.Random(2)
    for _ in range(100):
        n = rnd.randint(1, 10)
        graph, _ = random_weighted_graph(rnd, n, rnd.randint(0, 25), 1, 9)
        all_pairs = floyd_warshall(graph)
        for s in range(n):
            assert dijkstra(graph, s)[0] == bellman_ford(graph, s) == all_pairs[s]
