# Pattern: sorting (as a tool, and custom orderings)

**Signal:** order does not matter in the input but would make the problem easy; "arrange so
that ..."; custom comparison; counting inversions (merge sort); bounded integer keys.

## Representative LeetCode problems (names only)
- Sort an Array
- Sort Colors
- Largest Number
- Sort List
- Maximum Gap

## Solved example (my own wording)
**Problem.** Exam scores are whole numbers from 0 to 100. Sort a very large list of scores, and do
it faster than a comparison sort.

**Thinking.** There are only 101 possible values, so I do not need to compare scores at all:
count how many students got each score, then write each score out that many times. That is
counting sort: O(n + 101).

```python
def sort_scores(scores):
    counts = [0] * 101
    for s in scores:
        counts[s] += 1
    out = []
    for score, c in enumerate(counts):
        out.extend([score] * c)
    return out


sort_scores([88, 100, 71, 88, 0, 95])  # [0, 71, 88, 88, 95, 100]
```

**Complexity.** O(n + k) time and O(k) extra space for k possible values. (When records carry
other data and must stay stable, use the prefix-sum version in `impl.py`.)
