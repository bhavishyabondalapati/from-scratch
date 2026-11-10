# Graph traversal: BFS, DFS, topological sort, bipartite check

## Intuition
- **BFS** explores like ripples in a pond: everything 1 step away, then 2 steps, and so on. It
  uses a queue. Because it reaches nodes in order of distance, it gives shortest paths when every
  edge costs the same.
- **DFS** explores like walking a maze with a ball of string: go as deep as possible, back up
  when stuck. It uses a stack (or recursion). It is the tool for "explore everything", cycle
  detection, and ordering by dependencies.
- **Topological sort** lines up the nodes of a directed acyclic graph so every edge points
  forward (prerequisites before courses).
- **Bipartite** means the nodes split into two teams with every edge going between teams;
  equivalently, the graph has no odd-length cycle.

## Core idea
- Kahn: repeatedly output a node with in-degree 0 and delete its outgoing edges. If nodes
  remain when none has in-degree 0, they form a cycle.
- DFS topo: a node finishes only after everything reachable from it has finished, so *reverse
  finishing order* is a valid ordering. A gray (in-progress) neighbour is a back edge = cycle.
- Bipartite: BFS 2-colouring; an edge between same-coloured nodes proves an odd cycle.

## Complexity
All of these are O(V + E) time and O(V) extra space with adjacency lists
(an adjacency matrix makes them O(V²)).

## Common pitfalls
- Marking nodes visited when **dequeued** instead of when enqueued → duplicates in the queue.
- Using DFS for unweighted shortest paths (it finds *a* path, not the shortest).
- Directed cycle detection with a single "visited" set: you need the gray/black distinction
  (a visited node that is already finished is not a cycle).
- Forgetting disconnected graphs: loop over all nodes as possible starts.
- Recursion depth in Python on long paths; use the iterative version.

## When to use vs. alternatives
- Unweighted shortest path, levels, "minimum number of moves" → BFS.
- Weighted shortest path → Dijkstra / Bellman-Ford (see `shortest_paths`).
- Connectivity that grows over time → union-find.
- Kahn vs DFS topo: Kahn is iterative and can produce the lexicographically smallest order with
  a heap; DFS gives finishing times useful for SCCs.

## Interview questions
1. **Why does BFS give shortest paths in unweighted graphs?** Nodes are dequeued in
   non-decreasing distance; the first time a node is reached is via a shortest route.
2. **How do you detect a cycle in a directed graph?** DFS with three colours (back edge to a
   gray node), or Kahn's algorithm leaving nodes unprocessed.
3. **How is that different for undirected graphs?** Any visited neighbour other than the parent
   is a cycle; or use union-find.
4. **When is a graph bipartite?** Iff it has no odd cycle; test with BFS/DFS 2-colouring.
5. **Is the topological order unique?** Only if there is a Hamiltonian path in the DAG (at
   every step exactly one node has in-degree 0).
