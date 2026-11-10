import random
from itertools import product

from .impl import (
    bfs_distances,
    bfs_order,
    connected_components,
    dfs_iterative,
    dfs_recursive,
    has_cycle_directed,
    is_bipartite,
    num_islands,
    shortest_path,
    topological_sort_dfs,
    topological_sort_kahn,
)


def random_graph(rnd, n, m, directed):
    graph = {i: [] for i in range(n)}
    for _ in range(m):
        a, b = rnd.randrange(n), rnd.randrange(n)
        if a != b and b not in graph[a]:
            graph[a].append(b)
            if not directed:
                graph[b].append(a)
    return graph


def reachability(graph):
    """Floyd-Warshall-style hop distances as an independent reference."""
    n, INF = len(graph), float("inf")
    d = [[0 if i == j else INF for j in range(n)] for i in range(n)]
    for u in graph:
        for v in graph[u]:
            d[u][v] = 1
    for k, i, j in product(range(n), repeat=3):
        d[i][j] = min(d[i][j], d[i][k] + d[k][j])
    return d


def test_bfs_distances_and_paths_match_floyd_warshall():
    rnd = random.Random(0)
    for _ in range(100):
        n = rnd.randint(1, 9)
        graph = random_graph(rnd, n, rnd.randint(0, 14), directed=True)
        ref = reachability(graph)
        for s in range(n):
            dist = bfs_distances(graph, s)
            assert dist == {t: ref[s][t] for t in range(n) if ref[s][t] != float("inf")}
            for t in range(n):
                path = shortest_path(graph, s, t)
                if ref[s][t] == float("inf"):
                    assert path is None
                else:
                    assert path[0] == s and path[-1] == t and len(path) - 1 == ref[s][t]
                    assert all(b in graph[a] for a, b in zip(path, path[1:]))


def test_bfs_visits_in_nondecreasing_distance():
    rnd = random.Random(1)
    graph = random_graph(rnd, 30, 60, directed=False)
    dist = bfs_distances(graph, 0)
    order = bfs_order(graph, 0)
    assert [dist[u] for u in order] == sorted(dist[u] for u in order)
    assert set(order) == set(dist)


def test_dfs_iterative_matches_recursive():
    rnd = random.Random(2)
    for _ in range(200):
        n = rnd.randint(1, 12)
        graph = random_graph(rnd, n, rnd.randint(0, 25), directed=rnd.random() < 0.5)
        assert dfs_iterative(graph, 0) == dfs_recursive(graph, 0)
    g = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
    assert dfs_recursive(g, "a") == ["a", "b", "d", "c"]
    assert bfs_order(g, "a") == ["a", "b", "c", "d"]


def test_connected_components():
    graph = {0: [1], 1: [0], 2: [], 3: [4], 4: [3, 5], 5: [4]}
    assert sorted(sorted(c) for c in connected_components(graph)) == [[0, 1], [2], [3, 4, 5]]


def test_topological_sorts_are_valid_and_detect_cycles():
    rnd = random.Random(3)
    for _ in range(300):
        n = rnd.randint(1, 10)
        graph = random_graph(rnd, n, rnd.randint(0, 15), directed=True)
        ref = reachability(graph)
        cyclic = any(
            ref[u][v] != float("inf") and ref[v][u] != float("inf")
            for u in range(n)
            for v in range(n)
            if u != v
        )
        assert has_cycle_directed(graph) == cyclic
        for order in (topological_sort_kahn(graph), topological_sort_dfs(graph)):
            if cyclic:
                assert order is None
            else:
                position = {u: i for i, u in enumerate(order)}
                assert sorted(order) == list(range(n))
                assert all(position[u] < position[v] for u in graph for v in graph[u])


def test_is_bipartite_against_brute_force_colouring():
    rnd = random.Random(4)
    for _ in range(200):
        n = rnd.randint(1, 8)
        graph = random_graph(rnd, n, rnd.randint(0, 10), directed=False)
        expected = any(
            all(colours[u] != colours[v] for u in graph for v in graph[u])
            for colours in product([0, 1], repeat=n)
        )
        assert is_bipartite(graph) == expected


def test_num_islands():
    grid = [
        [1, 1, 0, 0, 0],
        [1, 0, 0, 1, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 1],
    ]
    assert num_islands(grid) == 4
    assert num_islands([]) == 0 and num_islands([[0]]) == 0 and num_islands([[1]]) == 1
