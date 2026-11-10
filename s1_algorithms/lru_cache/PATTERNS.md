# Pattern: design a data structure (hash map + linked list)

**Signal:** "design ... with O(1) operations", eviction policies, "most recent", ordering that
changes on access.

## Representative LeetCode problems (names only)
- LRU Cache
- Design Linked List
- Design Browser History
- All O`one Data Structure
- Insert Delete GetRandom O(1)

## Solved example (my own wording)
**Problem.** A phone shows the 3 most recently opened apps. Opening an app moves it to the front;
if a fourth app is opened, the one opened longest ago drops off. Support `open(app)` and
`recent()` efficiently.

**Thinking.** This is an LRU cache whose values do not matter. Python's `OrderedDict` is already
"hash map + doubly linked list", so the short version is below; the long version (my own nodes)
is in `impl.py`.

```python
from collections import OrderedDict


class RecentApps:
    def __init__(self, capacity=3):
        self.capacity, self.apps = capacity, OrderedDict()

    def open(self, app):
        self.apps.pop(app, None)  # if already present, remove so it re-enters at the end
        self.apps[app] = True
        if len(self.apps) > self.capacity:
            self.apps.popitem(last=False)  # drop the oldest

    def recent(self):
        return list(reversed(self.apps))


r = RecentApps()
for a in ["mail", "maps", "music", "mail", "camera"]:
    r.open(a)
r.recent()  # ['camera', 'mail', 'music']
```

**Complexity.** O(1) per `open`.
