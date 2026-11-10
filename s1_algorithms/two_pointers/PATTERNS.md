# Pattern: two pointers

**Signal:** sorted array, "pair/triplet with sum", "in place", palindromes, "partition".

## Representative LeetCode problems (names only)
- Valid Palindrome
- Two Sum II - Input Array Is Sorted
- 3Sum
- Container With Most Water
- Trapping Rain Water

## Solved example (my own wording)
**Problem.** Two hikers stand at opposite ends of a sorted list of trail marker distances. Find
two markers whose distances add up to exactly `d` kilometres, or report there are none.

**Thinking.** The list is sorted. Take the smallest and largest. If their sum is too small, the
smallest marker is useless (it is already paired with the largest possible partner), so drop it.
If the sum is too big, the largest is useless. Each step discards one marker.

```python
def marker_pair(dist, d):
    i, j = 0, len(dist) - 1
    while i < j:
        s = dist[i] + dist[j]
        if s == d:
            return dist[i], dist[j]
        if s < d:
            i += 1
        else:
            j -= 1
    return None


marker_pair([1, 3, 4, 7, 11], 11)  # (4, 7)
```

**Complexity.** O(n) time, O(1) space.
