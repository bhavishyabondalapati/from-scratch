"""Minimum spanning tree: Prim and Kruskal.

Input is an undirected graph given as n nodes and a list of (u, v, weight) edges.
Both functions return (total weight, chosen edges) for a minimum spanning FOREST,
so they also work when the graph is disconnected.
"""

from __future__ import annotations

import heapq

Edge = tuple[int, int, float]


def kruskal(n: int, edges: list[Edge]) -> tuple[float, list[Edge]]:
    """Sort edges by weight; keep an edge iff it joins two different components.

    O(E log E) for the sort; the union-find checks are near O(1) each.
    """
    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    total, chosen = 0.0, []
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        ru, rv = find(u), find(v)
        if ru == rv:
            continue  # would create a cycle
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        parent[rv] = ru  # union by size
        size[ru] += size[rv]
        total += w
        chosen.append((u, v, w))
    return total, chosen


def prim(n: int, edges: list[Edge]) -> tuple[float, list[Edge]]:
    """Grow a tree from a start node, always adding the cheapest edge leaving it.

    O(E log E) with a binary heap and lazy deletion. The outer loop restarts
    from every unvisited node so disconnected graphs give a spanning forest.
    """
    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    in_tree = [False] * n
    total, chosen = 0.0, []
    for start in range(n):
        if in_tree[start]:
            continue
        in_tree[start] = True
        heap = [(w, start, v) for v, w in adj[start]]  # edges crossing the "cut"
        heapq.heapify(heap)
        while heap:
            w, u, v = heapq.heappop(heap)
            if in_tree[v]:
                continue  # both ends already inside the tree: skip (lazy deletion)
            in_tree[v] = True
            total += w
            chosen.append((u, v, w))
            for nxt, nw in adj[v]:
                if not in_tree[nxt]:
                    heapq.heappush(heap, (nw, v, nxt))
    return total, chosen
