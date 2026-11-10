# Greedy + intervals

## Intuition
A greedy algorithm makes the choice that looks best *right now* and never reconsiders. It is fast
and simple, but only correct when you can argue that the local choice is always part of some
optimal answer. Interval problems are the classic home of greedy: sort the intervals (by start or
by end), then sweep once.

Rule of thumb for which key to sort by:
- **Merging / inserting / sweeping coverage** → sort by **start**.
- **Selecting the most non-overlapping / fewest points to hit all** → sort by **end**.

## Core idea — proving a greedy is right
**Exchange argument:** take any optimal solution; show you can swap its first choice for the
greedy choice without making it worse. Repeat down the line.

Example (activity selection): let `g` be the interval that ends first. In any optimal set, the
first interval `o` ends no earlier than `g`, so replacing `o` by `g` conflicts with nothing that
follows. So some optimal solution contains `g`.

**"Greedy stays ahead"** is the other common proof: after each step the greedy solution is at
least as good as any other partial solution (jump game: `farthest` reachable index).

## Complexity
| Problem | Time | Space |
|---|---|---|
| merge, max non-overlapping, min arrows, fractional knapsack | O(n log n) (the sort) | O(n) |
| insert into sorted intervals | O(n) | O(n) |
| meeting rooms (heap) | O(n log n) | O(n) |
| jump game I/II, gas station | O(n) | O(1) |

## Common pitfalls
- Applying greedy where it is not valid (0/1 knapsack, coin change with odd denominations).
- Sorting by the wrong key: earliest *start* or *shortest* interval both fail for activity selection.
- `<` vs `<=` at touching endpoints: decide whether `(1, 3)` and `(3, 5)` overlap for *this* problem.
- Meeting rooms: popping every finished meeting is unnecessary; reusing one room per arrival is enough.
- Gas station: forgetting the global feasibility check `sum(gas) >= sum(cost)`.

## When to use vs. alternatives
- Greedy when an exchange argument exists: scheduling, Huffman coding, MST, Dijkstra.
- DP when choices interact (weighted interval scheduling needs DP + binary search).
- Sweep line with events (+1 at start, −1 at end) as an alternative for "max overlap".

## Interview questions
1. **Why sort by end time for activity selection?** Finishing earliest leaves the most room; the
   exchange argument shows it is always safe.
2. **Give a counterexample for "shortest interval first".** `(0,5), (4,7), (6,11)`: the shortest
   one, `(4,7)`, blocks both others; optimal picks two.
3. **How do you compute the minimum number of meeting rooms?** Max number of simultaneously
   active intervals: min-heap of end times, or a sweep over sorted start/end events.
4. **How do you know a greedy algorithm is correct?** Exchange argument or stays-ahead proof;
   otherwise test against brute force and look for counterexamples.
5. **Why is fractional knapsack greedy but 0/1 knapsack not?** With fractions you can always fill
   remaining capacity with the best ratio; without them leftover capacity may be wasted.
