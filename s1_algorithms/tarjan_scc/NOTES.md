# Tarjan's strongly connected components

## Intuition
In a directed graph, a **strongly connected component** (SCC) is a group of nodes where everyone
can reach everyone else (think: a set of web pages that all link around to each other). Shrinking
each SCC to a single dot always leaves a DAG, which is why SCCs are the first step for many
directed-graph problems.

Tarjan finds them in one DFS. As DFS walks, it numbers nodes in discovery order and asks of each
node: "what is the *oldest* still-open node my subtree can get back to?" If the answer is the
node itself, then nothing below it escapes upward, and everything left on the stack above it forms
one complete SCC.

## Core idea
- `index[u]` = discovery time; `low[u]` = min discovery time reachable from u's subtree via at
  most one back/cross edge to a node **still on the stack**.
- Tree edge `u → v`: after returning, `low[u] = min(low[u], low[v])`.
- Edge to a node on the stack: `low[u] = min(low[u], index[v])`.
- Edge to a node already popped: ignore (that SCC is finished; it cannot reach back or it would
  not have been popped).
- `low[u] == index[u]` ⇒ pop the stack down to `u`: that is an SCC.
- **Bridges** (undirected): tree edge `(u, v)` is a bridge iff `low[v] > index[u]`.

## Complexity
O(V + E) time, O(V) space. One DFS (Kosaraju needs two passes and the reversed graph).

## Common pitfalls
- Updating `low` from nodes that are visited but **not on the stack** → merges separate SCCs.
- Popping the stack at the wrong time (must be after all of `u`'s edges are processed).
- Bridges: skipping the *parent node* rather than the *parent edge* breaks on parallel edges.
- Deep recursion in Python on big graphs (raise the limit or convert to an explicit stack).

## When to use vs. alternatives
- Tarjan: one pass, also yields reverse topological order of the SCC DAG.
- Kosaraju: easier to remember (DFS for finish order, then DFS on the reversed graph).
- Applications: 2-SAT, deadlock detection, finding cyclic dependency groups, dominance in
  "who can reach whom" queries, critical network links (bridges / articulation points).

## Interview questions
1. **What is an SCC and why is the condensation a DAG?** A maximal mutually reachable set; if two
   SCCs lay on a cycle they could reach each other and would be one SCC.
2. **What does `low[u] == index[u]` mean?** No node in u's subtree can reach an earlier node on
   the stack, so u is the first-discovered node of its SCC.
3. **Why must cross edges to finished SCCs be ignored?** Those nodes cannot reach back to the
   current path, so using their index would wrongly lower `low`.
4. **Tarjan vs Kosaraju?** Same O(V + E); Tarjan uses one DFS and no reversed graph.
5. **How do SCCs solve 2-SAT?** Build the implication graph; the formula is satisfiable iff no
   variable shares an SCC with its negation.
