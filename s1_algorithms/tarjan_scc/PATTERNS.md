# Pattern: strongly connected components / low-link

**Signal:** directed graph + "mutually reachable", "groups of cyclic dependencies", "critical
connections" (bridges), 2-SAT.

## Representative LeetCode problems (names only)
- Critical Connections in a Network
- Course Schedule
- Maximum Employees to Be Invited to a Meeting
- Minimum Number of Days to Disconnect Island
- Longest Cycle in a Graph

## Solved example (my own wording)
**Problem.** A build system has modules that import each other. Modules that import each other in
a loop must be compiled together as one unit. Group the modules into such units.

**Thinking.** "A imports B" is a directed edge. Modules that can all reach each other through
imports form a strongly connected component: exactly the units we want. Tarjan's algorithm
(`impl.py`) returns them, and conveniently in an order where each unit appears *after* the units
it depends on, which is also a valid build order.

```python
from s1_algorithms.tarjan_scc.impl import tarjan_scc

modules = ["app", "ui", "core", "utils"]
imports = {"app": ["ui"], "ui": ["core"], "core": ["ui", "utils"], "utils": []}

ids = {m: i for i, m in enumerate(modules)}
graph = [[ids[d] for d in imports[m]] for m in modules]
units = [[modules[i] for i in comp] for comp in tarjan_scc(graph)]
# [['utils'], ['core', 'ui'], ['app']]  -> build utils, then {core, ui} together, then app
```

**Complexity.** O(modules + imports).
