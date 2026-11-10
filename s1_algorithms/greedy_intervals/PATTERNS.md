# Pattern: greedy / intervals

**Signal:** lists of (start, end); "merge", "minimum rooms", "maximum events you can attend",
"fewest removals"; "can you reach the end"; a local rule that never needs undoing.

## Representative LeetCode problems (names only)
- Merge Intervals
- Non-overlapping Intervals
- Meeting Rooms II
- Minimum Number of Arrows to Burst Balloons
- Jump Game II

## Solved example (my own wording)
**Problem.** A single telescope can observe one event at a time. Each requested observation has a
start and end time. What is the largest number of requests that can be granted?

**Thinking.** Whichever request finishes first frees the telescope soonest, so granting it can
never hurt. Sort by end time and grant each request that starts after the last granted one ended.

```python
def most_observations(requests):
    granted, free_at = 0, float("-inf")
    for start, end in sorted(requests, key=lambda r: r[1]):
        if start >= free_at:
            granted += 1
            free_at = end
    return granted


most_observations([(1, 4), (3, 5), (0, 6), (5, 7), (8, 9), (5, 9)])  # 3: (1,4), (5,7), (8,9)
```

**Complexity.** O(n log n) for the sort, then one pass.
