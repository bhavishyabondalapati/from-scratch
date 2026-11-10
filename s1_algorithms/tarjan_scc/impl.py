"""Tarjan's algorithm for strongly connected components (plus bridges)."""

from __future__ import annotations


def tarjan_scc(graph: list[list[int]]) -> list[list[int]]:
    """Strongly connected components of a directed graph, in one DFS. O(V + E).

    An SCC is a maximal set of nodes that can all reach each other.

    For each node we record:
    - index[u]: the order in which DFS first discovered u;
    - low[u]:   the smallest index reachable from u's DFS subtree using tree
                edges plus at most one edge back to a node still on the stack.
    Nodes stay on an explicit stack until their SCC is complete. If, after
    exploring u, low[u] == index[u], then u cannot reach anything discovered
    earlier, so u is the "root" of an SCC: pop the stack down to u.

    Components come out in REVERSE topological order of the condensed graph
    (a component is emitted only after every component it can reach).
    """
    n = len(graph)
    index = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack: list[int] = []
    components: list[list[int]] = []
    counter = 0

    def visit(u: int) -> None:
        nonlocal counter
        index[u] = low[u] = counter
        counter += 1
        stack.append(u)
        on_stack[u] = True
        for v in graph[u]:
            if index[v] == -1:  # tree edge: explore, then inherit v's low
                visit(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:  # edge back into the current, unfinished SCC
                low[u] = min(low[u], index[v])
            # else: v belongs to an SCC that is already complete; ignore it
        if low[u] == index[u]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == u:
                    break
            components.append(component)

    for u in range(n):
        if index[u] == -1:
            visit(u)
    return components


def condensation(graph: list[list[int]]) -> tuple[list[int], list[set[int]]]:
    """Collapse each SCC to one node. Returns (component id per node, DAG adjacency)."""
    components = tarjan_scc(graph)
    comp_of = [0] * len(graph)
    for cid, comp in enumerate(components):
        for u in comp:
            comp_of[u] = cid
    dag: list[set[int]] = [set() for _ in components]
    for u, neighbours in enumerate(graph):
        for v in neighbours:
            if comp_of[u] != comp_of[v]:
                dag[comp_of[u]].add(comp_of[v])
    return comp_of, dag


def find_bridges(n: int, edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Edges of an UNDIRECTED graph whose removal disconnects it. O(V + E).

    Same low-link idea: tree edge (u, v) is a bridge iff nothing in v's
    subtree can climb back to u or above, i.e. low[v] > index[u].
    """
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for edge_id, (a, b) in enumerate(edges):
        adj[a].append((b, edge_id))
        adj[b].append((a, edge_id))
    index = [-1] * n
    low = [0] * n
    bridges: list[tuple[int, int]] = []
    counter = 0

    def visit(u: int, via_edge: int) -> None:
        nonlocal counter
        index[u] = low[u] = counter
        counter += 1
        for v, edge_id in adj[u]:
            if edge_id == via_edge:
                continue  # do not walk straight back along the edge we came in on
            if index[v] == -1:
                visit(v, edge_id)
                low[u] = min(low[u], low[v])
                if low[v] > index[u]:
                    bridges.append(edges[edge_id])
            else:
                low[u] = min(low[u], index[v])

    for u in range(n):
        if index[u] == -1:
            visit(u, -1)
    return bridges
