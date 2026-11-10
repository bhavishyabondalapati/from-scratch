"""Shortest paths: Dijkstra, Bellman-Ford, Floyd-Warshall.

Weighted graphs are adjacency lists: graph[u] = [(v, weight), ...] with nodes 0..n-1.
"""

from __future__ import annotations

import heapq

INF = float("inf")
WeightedGraph = list[list[tuple[int, float]]]


def dijkstra(graph: WeightedGraph, source: int) -> tuple[list[float], list[int | None]]:
    """Single-source shortest paths with NON-NEGATIVE weights. O((V + E) log V).

    Greedy idea: the unfinished node with the smallest tentative distance can
    never be improved later (all other routes are already at least as long and
    weights cannot be negative), so finalise it and relax its outgoing edges.

    Returns (dist, parent) where parent lets you rebuild the paths.
    """
    n = len(graph)
    dist: list[float] = [INF] * n
    parent: list[int | None] = [None] * n
    dist[source] = 0
    heap: list[tuple[float, int]] = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # stale entry: u was already finalised with a shorter distance
        for v, w in graph[u]:
            if w < 0:
                raise ValueError("Dijkstra requires non-negative weights")
            if d + w < dist[v]:  # "relaxation"
                dist[v] = d + w
                parent[v] = u
                heapq.heappush(heap, (dist[v], v))  # lazy insert instead of decrease-key
    return dist, parent


def reconstruct_path(parent: list[int | None], source: int, target: int) -> list[int] | None:
    """Follow parent pointers back from target; None if target is unreachable."""
    path = [target]
    while path[-1] != source:
        prev = parent[path[-1]]
        if prev is None:
            return None
        path.append(prev)
    return path[::-1]


def bellman_ford(graph: WeightedGraph, source: int) -> list[float] | None:
    """Single-source shortest paths allowing negative weights. O(V * E).

    A shortest path uses at most V - 1 edges, so relaxing every edge V - 1
    times is enough. If a V-th pass still improves something, a negative cycle
    is reachable from the source: return None.
    """
    n = len(graph)
    dist: list[float] = [INF] * n
    dist[source] = 0
    for _ in range(n - 1):
        changed = False
        for u in range(n):
            if dist[u] == INF:
                continue
            for v, w in graph[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    changed = True
        if not changed:  # early exit: nothing improved in a full pass
            break
    for u in range(n):
        if dist[u] != INF and any(dist[u] + w < dist[v] for v, w in graph[u]):
            return None
    return dist


def floyd_warshall(graph: WeightedGraph) -> list[list[float]] | None:
    """All-pairs shortest paths. O(V^3) time, O(V^2) space. None on a negative cycle.

    DP over allowed intermediate nodes: after iteration k, dist[i][j] is the
    shortest i -> j path that only passes through nodes 0..k. Either that path
    avoids k (keep the old value) or it goes i -> k -> j.
    """
    n = len(graph)
    dist = [[INF] * n for _ in range(n)]
    for u in range(n):
        dist[u][u] = 0
        for v, w in graph[u]:
            dist[u][v] = min(dist[u][v], w)
    for k in range(n):  # k MUST be the outermost loop
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            if dik == INF:
                continue
            di = dist[i]
            for j in range(n):
                if dik + dk[j] < di[j]:
                    di[j] = dik + dk[j]
    if any(dist[i][i] < 0 for i in range(n)):
        return None  # some node can reach itself with negative total weight
    return dist
