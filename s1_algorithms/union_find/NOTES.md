# Union-find (disjoint set union)

## Intuition
Keep track of which items belong to the same group when groups only ever **merge**. Each group
elects a representative ("root"). Two items are in the same group iff they have the same root.
Merging two groups = making one root point at the other.

## Core idea
- `find(x)`: follow parent pointers to the root.
- **Path compression:** after finding the root, re-point every node on the path directly at it,
  so the next `find` is one hop.
- **Union by rank (or size):** hang the shorter tree under the taller one, so trees stay shallow
  (height ≤ log₂ n even without compression).
- With both, m operations cost O(m · α(n)), where α is the inverse Ackermann function
  (≤ 4 for any n that fits in the universe) — effectively constant.

## Complexity
| Operation | Time | Space |
|---|---|---|
| find / union / connected | O(α(n)) amortized | O(n) |
| count components / detect cycle over E edges | O(E · α(n)) | O(n) |

## Common pitfalls
- Unioning the raw nodes instead of their **roots** (`parent[a] = b` is wrong).
- Forgetting to update size/rank only at the root.
- Expecting to *split* groups: union-find cannot undo merges (needs offline tricks / rollback DSU).
- Using it on **directed** reachability: it models undirected connectivity only.
- Recursive `find` hitting Python's recursion limit on long chains (iterative here).

## When to use vs. alternatives
- Edges arrive over time and you ask "connected yet?" → union-find.
- One-off components on a fixed graph → BFS/DFS is just as good.
- Kruskal's MST, cycle detection in undirected graphs, merging equivalent items, percolation.
- Need shortest paths or the actual path → BFS/DFS, not union-find.

## Interview questions
1. **What do path compression and union by rank each buy you?** Rank alone gives O(log n);
   compression alone gives O(log n) amortized; together O(α(n)).
2. **How do you detect a cycle in an undirected graph?** For each edge, if both ends already
   share a root the edge closes a cycle.
3. **Why does Kruskal use union-find?** To test in near-constant time whether an edge joins two
   different components.
4. **Can union-find delete an edge?** Not directly; process queries offline in reverse (deletions
   become unions) or use a rollback structure without path compression.
5. **Union-find vs DFS for connected components?** Same result on a static graph; union-find
   wins when the graph grows incrementally.
