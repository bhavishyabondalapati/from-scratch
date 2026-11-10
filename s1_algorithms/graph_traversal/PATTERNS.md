# Pattern: BFS / DFS / topological sort

**Signal:** grids and networks; "fewest steps"; "can you finish all courses"; "order of tasks";
"number of islands/regions"; "split into two groups".

## Representative LeetCode problems (names only)
- Number of Islands
- Course Schedule II
- Rotting Oranges
- Is Graph Bipartite?
- Word Ladder

## Solved example (my own wording)
**Problem.** Getting dressed has rules such as "socks before shoes" and "shirt before jacket".
Given the items and the rules, produce an order that obeys every rule, or say it is impossible.

**Thinking.** Treat each rule `a before b` as an edge `a -> b`. An item with no unmet "before"
rules can be put on now. Putting it on may free other items. If I ever get stuck with items left,
the rules contradict each other (a cycle).

```python
from collections import deque


def dressing_order(items, rules):
    after = {i: [] for i in items}
    waiting_on = dict.fromkeys(items, 0)
    for a, b in rules:
        after[a].append(b)
        waiting_on[b] += 1
    ready = deque(i for i in items if waiting_on[i] == 0)
    order = []
    while ready:
        item = ready.popleft()
        order.append(item)
        for nxt in after[item]:
            waiting_on[nxt] -= 1
            if waiting_on[nxt] == 0:
                ready.append(nxt)
    return order if len(order) == len(items) else None


dressing_order(
    ["shoes", "socks", "shirt", "jacket"],
    [("socks", "shoes"), ("shirt", "jacket")],
)  # ['socks', 'shirt', 'shoes', 'jacket']
```

**Complexity.** O(items + rules).
