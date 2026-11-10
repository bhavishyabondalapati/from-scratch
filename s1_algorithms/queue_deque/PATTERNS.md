# Pattern: queue / deque / monotonic queue

**Signal:** "maximum/minimum of every window of size k", level-by-level processing, "implement
queue using ...", shortest subarray when negatives are present.

## Representative LeetCode problems (names only)
- Implement Queue using Stacks
- Design Circular Deque
- Sliding Window Maximum
- Shortest Subarray with Sum at Least K
- Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

## Solved example (my own wording)
**Problem.** A weather station logs a temperature every minute. For a dashboard, report the
hottest reading within each rolling window of `k` minutes.

**Thinking.** Keep a deque of minutes that could still be "the hottest" for some future window.
A reading that is both older and cooler than the newest reading can never win again, so remove it
from the back before adding the new one. Then the front is always the hottest; drop it when it
is more than `k` minutes old.

```python
from collections import deque


def rolling_hottest(temps, k):
    dq, out = deque(), []  # indices, temperatures decreasing
    for i, t in enumerate(temps):
        while dq and temps[dq[-1]] <= t:
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            out.append(temps[dq[0]])
    return out


rolling_hottest([21, 23, 22, 20, 25, 24], 3)  # [23, 23, 25, 25]
```

**Complexity.** O(n) time, O(k) space.
