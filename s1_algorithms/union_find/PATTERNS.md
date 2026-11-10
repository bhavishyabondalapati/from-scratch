# Pattern: union-find

**Signal:** "are these connected?", "number of groups", "merge accounts/sets", "redundant
connection", edges added over time.

## Representative LeetCode problems (names only)
- Number of Provinces
- Redundant Connection
- Accounts Merge
- Number of Connected Components in an Undirected Graph
- Satisfiability of Equality Equations

## Solved example (my own wording)
**Problem.** Islands are being joined by bridges, one at a time. After each new bridge, report how
many separate groups of islands remain (a group = islands reachable from each other by bridges).

**Thinking.** Start with every island as its own group. A bridge between two islands in different
groups merges them (count goes down by one); a bridge inside one group changes nothing.

```python
def groups_after_each_bridge(n, bridges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    groups, out = n, []
    for a, b in bridges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            groups -= 1
        out.append(groups)
    return out


groups_after_each_bridge(5, [(0, 1), (2, 3), (1, 0), (1, 2)])  # [4, 3, 3, 2]
```

**Complexity.** Near O(1) per bridge.
