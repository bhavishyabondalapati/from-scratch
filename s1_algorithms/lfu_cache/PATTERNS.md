# Pattern: design with frequency buckets

**Signal:** "least/most frequently used", "increment/decrement a key and get the max/min key in
O(1)", any design where items move between adjacent count levels.

## Representative LeetCode problems (names only)
- LFU Cache
- All O`one Data Structure
- Maximum Frequency Stack
- Design a Leaderboard
- Top K Frequent Words

## Solved example (my own wording)
**Problem.** A vending machine has room for only `n` drink types. Each time a drink is bought its
popularity goes up by one. When a new drink type is introduced and the machine is full, remove the
drink bought the fewest times (if tied, the one whose last purchase was longest ago).

**Thinking.** Keep one shelf per purchase count. A purchase moves a drink from shelf `f` to shelf
`f + 1`, placing it at the newest end. Remember the lowest non-empty shelf; the drink to remove is
the oldest on that shelf.

```python
from collections import OrderedDict, defaultdict


class Machine:
    def __init__(self, n):
        self.n, self.count, self.low = n, {}, 0
        self.shelf = defaultdict(OrderedDict)

    def buy(self, drink):
        if drink not in self.count:
            return False
        f = self.count[drink]
        del self.shelf[f][drink]
        if not self.shelf[f] and self.low == f:
            self.low = f + 1
        self.count[drink] = f + 1
        self.shelf[f + 1][drink] = None
        return True

    def introduce(self, drink):
        if drink in self.count:
            return
        if len(self.count) == self.n:
            old, _ = self.shelf[self.low].popitem(last=False)
            del self.count[old]
        self.count[drink], self.low = 1, 1
        self.shelf[1][drink] = None
```

**Complexity.** O(1) per operation.
