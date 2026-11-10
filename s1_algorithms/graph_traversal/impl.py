"""Graph traversal: BFS, DFS, topological sort (Kahn + DFS), bipartite check.

Graphs are adjacency lists: dict[node, list[neighbour]]. Every node must be a key.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable

Graph = dict[Hashable, list[Hashable]]


def bfs_order(graph: Graph, start: Hashable) -> list[Hashable]:
    """Nodes in the order BFS visits them (closest first)."""
    seen, order, queue = {start}, [], deque([start])
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            if v not in seen:  # mark when ENQUEUED so nothing is queued twice
                seen.add(v)
                queue.append(v)
    return order


def bfs_distances(graph: Graph, start: Hashable) -> dict[Hashable, int]:
    """Fewest edges from start to every reachable node (unweighted shortest path)."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def shortest_path(graph: Graph, start: Hashable, goal: Hashable) -> list[Hashable] | None:
    """One shortest path (by edge count) from start to goal, or None."""
    parent: dict[Hashable, Hashable | None] = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u == goal:
            path = []
            while u is not None:  # walk parents back to the start
                path.append(u)
                u = parent[u]
            return path[::-1]
        for v in graph[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    return None


def dfs_recursive(graph: Graph, start: Hashable) -> list[Hashable]:
    """Preorder DFS: visit a node, then fully explore each neighbour in turn."""
    seen: set[Hashable] = set()
    order: list[Hashable] = []

    def visit(u: Hashable) -> None:
        seen.add(u)
        order.append(u)
        for v in graph[u]:
            if v not in seen:
                visit(v)

    visit(start)
    return order


def dfs_iterative(graph: Graph, start: Hashable) -> list[Hashable]:
    """Same visiting order as dfs_recursive, with an explicit stack."""
    seen: set[Hashable] = set()
    order: list[Hashable] = []
    stack = [start]
    while stack:
        u = stack.pop()
        if u in seen:  # a node can be pushed several times; visit it once
            continue
        seen.add(u)
        order.append(u)
        stack.extend(reversed(graph[u]))  # reversed so the first neighbour pops first
    return order


def connected_components(graph: Graph) -> list[list[Hashable]]:
    """Components of an undirected graph."""
    seen: set[Hashable] = set()
    components = []
    for s in graph:
        if s not in seen:
            component = bfs_order(graph, s)
            seen.update(component)
            components.append(component)
    return components


def topological_sort_kahn(graph: Graph) -> list[Hashable] | None:
    """Kahn's algorithm (BFS on in-degrees). Returns None if there is a cycle.

    A node with in-degree 0 has no unmet prerequisites, so it can go next.
    Removing it lowers its neighbours' in-degrees, possibly freeing them.
    """
    indegree = dict.fromkeys(graph, 0)
    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
    queue = deque(u for u in graph if indegree[u] == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                queue.append(v)
    return order if len(order) == len(graph) else None  # leftovers => cycle


def topological_sort_dfs(graph: Graph) -> list[Hashable] | None:
    """DFS-based topological sort: reverse of the order nodes FINISH.

    Three colours: white = unvisited, gray = on the current DFS path,
    black = finished. Meeting a gray node means we found a back edge: a cycle.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    colour = dict.fromkeys(graph, WHITE)
    finished: list[Hashable] = []

    def visit(u: Hashable) -> bool:
        colour[u] = GRAY
        for v in graph[u]:
            if colour[v] == GRAY or (colour[v] == WHITE and not visit(v)):
                return False
        colour[u] = BLACK
        finished.append(u)  # all descendants are done, so u can come before them
        return True

    for u in graph:
        if colour[u] == WHITE and not visit(u):
            return None
    return finished[::-1]


def has_cycle_directed(graph: Graph) -> bool:
    return topological_sort_kahn(graph) is None


def is_bipartite(graph: Graph) -> bool:
    """Can the nodes of an undirected graph be 2-coloured with no same-colour edge?

    BFS, giving each neighbour the opposite colour. A conflict means an odd cycle.
    """
    colour: dict[Hashable, int] = {}
    for s in graph:
        if s in colour:
            continue
        colour[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in colour:
                    colour[v] = 1 - colour[u]
                    queue.append(v)
                elif colour[v] == colour[u]:
                    return False
    return True


def num_islands(grid: list[list[int]]) -> int:
    """Count groups of 1-cells connected up/down/left/right (grid as an implicit graph)."""
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    islands = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1 and (r, c) not in seen:
                islands += 1
                seen.add((r, c))
                stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if (
                            0 <= nr < rows
                            and 0 <= nc < cols
                            and grid[nr][nc] == 1
                            and (nr, nc) not in seen
                        ):
                            seen.add((nr, nc))
                            stack.append((nr, nc))
    return islands
