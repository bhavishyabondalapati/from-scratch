# Minimum spanning tree: Prim and Kruskal

## Intuition
Connect all the towns with roads as cheaply as possible. The result has no cycles (a cycle means
one road is redundant), so it is a tree with exactly V − 1 edges.

- **Kruskal:** look at roads from cheapest to most expensive; build a road unless the two towns
  are already connected. Thinks in terms of *edges* and merges components (union-find).
- **Prim:** start in one town and repeatedly build the cheapest road from the connected region to
  a new town. Thinks in terms of *growing one tree* (heap).

## Core math
**Cut property:** split the nodes into two non-empty groups; the lightest edge crossing the split
belongs to some MST. (If an MST did not use it, adding it creates a cycle that crosses the cut a
second time through a heavier-or-equal edge; swapping them does not increase the weight.) Both
algorithms repeatedly apply this property with different cuts.

**Cycle property:** the heaviest edge on any cycle is not needed.

## Complexity
| Algorithm | Time | Space | Best for |
|---|---|---|---|
| Kruskal | O(E log E) | O(V + E) | sparse graphs, edge lists |
| Prim (binary heap) | O(E log V) | O(V + E) | dense graphs, adjacency lists |
| Prim (array, no heap) | O(V²) | O(V) | very dense graphs |

## Common pitfalls
- Confusing MST with shortest paths: the MST minimises *total* weight, not the distance between
  any particular pair.
- Prim: not skipping heap entries whose endpoint is already in the tree.
- Disconnected graphs: there is no spanning tree, only a spanning forest.
- MSTs are for **undirected** graphs (the directed analogue is the harder arborescence problem).
- The MST is unique only if all edge weights are distinct.

## When to use vs. alternatives
- Network design, clustering (remove the k − 1 heaviest MST edges → k clusters, which is
  single-linkage clustering), approximating TSP.
- Edges already sorted or easy to sort → Kruskal. Dense graph or implicit edges (points in a
  plane) → Prim.

## Interview questions
1. **Why is the greedy choice safe?** The cut property: the cheapest edge across any cut is in
   some MST.
2. **Prim vs Kruskal?** Same result; Kruskal sorts edges and uses union-find, Prim grows one
   component with a heap. Prim's O(V²) form wins on dense graphs.
3. **Do negative weights break them?** No; only relative order matters (unlike Dijkstra).
4. **How do Prim and Dijkstra differ?** Same skeleton; Prim's key is the single edge weight into
   the tree, Dijkstra's is the total distance from the source.
5. **How would you find the second-best MST or check uniqueness?** For each non-tree edge, add
   it and remove the heaviest edge on the cycle it forms; take the best such swap.
