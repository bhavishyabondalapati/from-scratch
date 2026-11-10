# Shortest paths: Dijkstra, Bellman-Ford, Floyd-Warshall

## Intuition
All three are built on one move, **relaxation**: "if going to `v` through `u` is shorter than the
best route to `v` I know, update it". They differ in the *order* they relax edges.

- **Dijkstra** (greedy): always finalise the closest unfinished node. Correct only when weights
  are non-negative, because then a finalised distance can never be beaten later.
- **Bellman-Ford** (brute-force DP): relax *every* edge, V − 1 times. Slower, but handles
  negative edges and detects negative cycles.
- **Floyd-Warshall** (DP over intermediates): all pairs at once; allow node 0 as a stopover,
  then node 1, ...

## Core math
- Relaxation: `dist[v] = min(dist[v], dist[u] + w(u, v))`.
- Bellman-Ford invariant: after `i` passes, `dist[v]` ≤ the shortest path using at most `i`
  edges. Shortest simple paths have ≤ V − 1 edges.
- Floyd-Warshall recurrence: `d_k[i][j] = min(d_{k-1}[i][j], d_{k-1}[i][k] + d_{k-1}[k][j])`.

## Complexity
| Algorithm | Time | Space | Negative edges | Scope |
|---|---|---|---|---|
| BFS (unweighted) | O(V + E) | O(V) | n/a | single source |
| Dijkstra (binary heap) | O((V + E) log V) | O(V + E) | no | single source |
| Bellman-Ford | O(V · E) | O(V) | yes, detects negative cycles | single source |
| Floyd-Warshall | O(V³) | O(V²) | yes, detects negative cycles | all pairs |

## Common pitfalls
- Running Dijkstra with negative edges: silently wrong answers.
- Not skipping stale heap entries (`if d > dist[u]: continue`) → correct but much slower.
- Floyd-Warshall with `k` not as the outermost loop → wrong.
- Bellman-Ford: relaxing from nodes whose distance is still infinity; forgetting that only
  negative cycles *reachable from the source* matter.
- `inf + negative` arithmetic in languages with integer infinity sentinels.

## When to use vs. alternatives
- Unweighted → BFS. Weights 0/1 → 0-1 BFS with a deque.
- Non-negative weights → Dijkstra. With a good heuristic toward one target → A*.
- Negative weights or need cycle detection (e.g. currency arbitrage) → Bellman-Ford.
- Dense graph, all pairs, small V (≤ ~400) → Floyd-Warshall. Sparse all-pairs → Dijkstra from
  every node (or Johnson's algorithm with negative edges).
- DAG → relax in topological order, O(V + E), any weights.

## Interview questions
1. **Why does Dijkstra fail with negative edges?** A node may be finalised and later a cheaper
   route through a negative edge appears; the greedy choice is no longer safe.
2. **Why V − 1 passes in Bellman-Ford?** A shortest path without cycles has at most V − 1 edges
   and each pass extends correct prefixes by at least one edge.
3. **How do you detect a negative cycle?** If any edge can still be relaxed after V − 1 passes
   (Bellman-Ford), or some `dist[i][i] < 0` (Floyd-Warshall).
4. **Dijkstra's complexity without decrease-key?** With lazy deletion the heap holds up to E
   entries: O(E log E) = O(E log V).
5. **How would you find the path, not just the distance?** Store a parent pointer whenever a
   relaxation succeeds and walk back from the target.
