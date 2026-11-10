# Pattern: Fenwick tree

**Signal:** running counts or sums that change over time; "how many smaller/larger before me";
inversions; rank queries.

## Representative LeetCode problems (names only)
- Range Sum Query - Mutable
- Count of Smaller Numbers After Self
- Reverse Pairs
- Create Sorted Array through Instructions
- Queue Reconstruction by Height

## Solved example (my own wording)
**Problem.** Runners cross a finish line one by one; each has a bib number from 1 to `m`. As each
runner finishes, report how many earlier finishers had a *higher* bib number.

**Thinking.** Keep a Fenwick tree over bib numbers where position `b` holds 1 if bib `b` has
finished. "Earlier finishers with a higher bib" = (number finished so far) − (finished with
bib ≤ mine), and the second part is a prefix sum.

```python
def higher_bibs_before(bibs, m):
    tree = [0] * (m + 1)

    def add(i):
        while i <= m:
            tree[i] += 1
            i += i & -i

    def prefix(i):
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    out = []
    for finished, b in enumerate(bibs):
        out.append(finished - prefix(b))
        add(b)
    return out


higher_bibs_before([3, 1, 4, 2], 4)  # [0, 1, 0, 2]
```

**Complexity.** O(n log m). The total of the output is the number of inversions.
