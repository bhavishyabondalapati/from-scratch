# Pattern: heap / top-k / two heaps / k-way merge

**Signal:** "k largest/smallest/closest/most frequent", "merge k sorted ...", "running median",
"always process the cheapest/earliest next".

## Representative LeetCode problems (names only)
- Kth Largest Element in an Array
- Merge k Sorted Lists
- Find Median from Data Stream
- K Closest Points to Origin
- Task Scheduler

## Solved example (my own wording)
**Problem.** Scores arrive one at a time during a long tournament. At any moment, report the
scores of the current top 3 players without storing every score.

**Thinking.** Keep only three scores in a min-heap. The smallest of those three is at the root:
it is the score a newcomer must beat. If the new score beats it, replace it.

```python
import heapq


def top3_stream(scores):
    podium = []  # min-heap of at most 3 scores
    for s in scores:
        if len(podium) < 3:
            heapq.heappush(podium, s)
        elif s > podium[0]:
            heapq.heapreplace(podium, s)  # pop the smallest, push s
        yield sorted(podium, reverse=True)


list(top3_stream([50, 70, 20, 90, 60]))[-1]  # [90, 70, 60]
```

**Complexity.** O(log 3) = O(1) per score here; O(n log k) in general, O(k) memory.
