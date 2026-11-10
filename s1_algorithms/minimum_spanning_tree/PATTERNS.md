# Pattern: minimum spanning tree

**Signal:** "connect all points/cities with minimum total cost", "cheapest network", "critical
edges".

## Representative LeetCode problems (names only)
- Min Cost to Connect All Points
- Connecting Cities With Minimum Cost
- Optimize Water Distribution in a Village
- Find Critical and Pseudo-Critical Edges in Minimum Spanning Tree
- Remove Max Number of Edges to Keep Graph Fully Traversable

## Solved example (my own wording)
**Problem.** A campus has buildings at given (x, y) positions. Laying fibre between two buildings
costs their Manhattan distance. What is the cheapest way to connect every building?

**Thinking.** Every pair of buildings is a possible edge, so the graph is dense (n² edges). Prim
without a heap suits this: keep, for each building not yet connected, the cheapest cost to attach
it to the connected set; repeatedly attach the cheapest one and update the others.

```python
def cheapest_fibre(points):
    n = len(points)
    attach_cost = [float("inf")] * n
    attach_cost[0] = 0
    connected = [False] * n
    total = 0
    for _ in range(n):
        u = min((i for i in range(n) if not connected[i]), key=attach_cost.__getitem__)
        connected[u] = True
        total += attach_cost[u]
        for v in range(n):
            if not connected[v]:
                d = abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1])
                attach_cost[v] = min(attach_cost[v], d)
    return total


cheapest_fibre([(0, 0), (2, 2), (3, 10), (5, 2), (7, 0)])  # 20
```

**Complexity.** O(n²) time, O(n) space.
