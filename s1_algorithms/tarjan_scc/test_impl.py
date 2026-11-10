import random

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .impl import condensation, find_bridges, tarjan_scc


def random_digraph(rnd, n, m):
    graph = [[] for _ in range(n)]
    for _ in range(m):
        a, b = rnd.randrange(n), rnd.randrange(n)
        if b not in graph[a]:
            graph[a].append(b)  # self-loops allowed
    return graph


def to_csr(graph):
    rows = [u for u, nbrs in enumerate(graph) for _ in nbrs]
    cols = [v for nbrs in graph for v in nbrs]
    return csr_matrix(([1] * len(rows), (rows, cols)), shape=(len(graph), len(graph)))


def test_scc_matches_scipy():
    rnd = random.Random(0)
    for _ in range(300):
        n = rnd.randint(1, 14)
        graph = random_digraph(rnd, n, rnd.randint(0, 25))
        components = tarjan_scc(graph)
        assert sorted(u for c in components for u in c) == list(range(n))  # a partition
        k, labels = connected_components(to_csr(graph), directed=True, connection="strong")
        assert len(components) == k
        for comp in components:
            assert len({labels[u] for u in comp}) == 1


def test_known_example():
    # 0 -> 1 -> 2 -> 0 is a cycle; 3 -> 4 -> 5 -> 3 is a cycle; 2 -> 3 links them; 6 alone.
    graph = [[1], [2], [0, 3], [4], [5], [3], []]
    comps = sorted(sorted(c) for c in tarjan_scc(graph))
    assert comps == [[0, 1, 2], [3, 4, 5], [6]]


def test_components_come_out_in_reverse_topological_order():
    rnd = random.Random(1)
    for _ in range(200):
        graph = random_digraph(rnd, rnd.randint(1, 12), rnd.randint(0, 20))
        comp_of, dag = condensation(graph)
        # Every edge of the condensed DAG goes from a later-emitted to an earlier-emitted SCC.
        for a, targets in enumerate(dag):
            assert all(b < a for b in targets)
        assert all(comp_of[u] >= comp_of[v] for u in range(len(graph)) for v in graph[u])


def n_components(n, es):
    rows, cols = [e[0] for e in es], [e[1] for e in es]
    matrix = csr_matrix(([1] * len(es), (rows, cols)), shape=(n, n))
    return connected_components(matrix, directed=False)[0]


def test_bridges_against_brute_force():
    rnd = random.Random(2)
    for _ in range(200):
        n = rnd.randint(1, 10)
        edges = list(
            {tuple(sorted((rnd.randrange(n), rnd.randrange(n)))) for _ in range(rnd.randint(0, 14))}
        )
        edges = [(a, b) for a, b in edges if a != b]

        base = n_components(n, edges)
        expected = sorted(e for e in edges if n_components(n, [x for x in edges if x != e]) > base)
        assert sorted(find_bridges(n, edges)) == expected


def test_parallel_edges_are_not_bridges():
    assert find_bridges(2, [(0, 1), (0, 1)]) == []
    assert find_bridges(2, [(0, 1)]) == [(0, 1)]
