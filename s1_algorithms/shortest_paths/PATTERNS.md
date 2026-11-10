# Pattern: weighted shortest paths

**Signal:** "minimum cost/time to reach", weighted edges, "at most k stops", "cheapest",
"arbitrage" (negative cycle).

## Representative LeetCode problems (names only)
- Network Delay Time
- Cheapest Flights Within K Stops
- Path With Minimum Effort
- Find the City With the Smallest Number of Neighbors at a Threshold Distance
- Swim in Rising Water

## Solved example (my own wording)
**Problem.** A message starts at one server and spreads along one-way links, each with a delay in
milliseconds. How long until *every* server has the message? Return -1 if some server never gets it.

**Thinking.** Each server first hears the message via its shortest-delay route from the source, so
I need single-source shortest paths with non-negative weights: Dijkstra. The answer is the largest
of those shortest delays.

```python
import heapq


def time_to_reach_all(n, links, source):
    graph = [[] for _ in range(n)]
    for a, b, delay in links:
        graph[a].append((b, delay))
    best = [float("inf")] * n
    best[source] = 0
    heap = [(0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > best[u]:
            continue
        for v, w in graph[u]:
            if d + w < best[v]:
                best[v] = d + w
                heapq.heappush(heap, (best[v], v))
    slowest = max(best)
    return -1 if slowest == float("inf") else slowest


time_to_reach_all(4, [(0, 1, 5), (0, 2, 1), (2, 1, 2), (1, 3, 4)], 0)  # 7
```

**Complexity.** O((V + E) log V).
