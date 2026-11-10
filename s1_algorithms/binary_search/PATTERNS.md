# Pattern: binary search (including on the answer space)

**Signal:** sorted input; "find first/last position"; O(log n) required; "minimum capacity /
speed / largest sum such that ..." with a monotonic feasibility check.

## Representative LeetCode problems (names only)
- Find First and Last Position of Element in Sorted Array
- Search in Rotated Sorted Array
- Find Minimum in Rotated Sorted Array
- Koko Eating Bananas
- Split Array Largest Sum

## Solved example (my own wording)
**Problem.** A print shop has a queue of jobs with given page counts and must finish them within
`h` hours. The printer handles one job at a time and, in each hour, prints at most `s` pages of
the current job (a job that finishes early does not let the next job start until the next hour).
What is the slowest printer speed `s` that still finishes on time?

**Thinking.** If speed `s` is fast enough, any faster speed also is. So "is `s` enough?" looks like
`No No No Yes Yes` over `s = 1 .. max(pages)`, and I want the first Yes. Checking one speed is a
single pass: job `p` takes `ceil(p / s)` hours.

```python
import math


def slowest_speed(pages, h):
    lo, hi = 1, max(pages)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(math.ceil(p / mid) for p in pages) <= h:
            hi = mid  # fast enough, try slower
        else:
            lo = mid + 1
    return lo


slowest_speed([30, 11, 23, 4, 20], 6)  # 23
```

**Complexity.** O(n log(max pages)).
