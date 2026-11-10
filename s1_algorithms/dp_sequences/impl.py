"""Sequence DP: longest increasing subsequence, longest common subsequence, edit distance."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence


def lis_quadratic(nums: Sequence[int]) -> int:
    """LIS length in O(n^2). best[i] = length of the longest increasing
    subsequence that ENDS at index i."""
    best = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best, default=0)


def lis_length(nums: Sequence[int]) -> int:
    """LIS length in O(n log n) with "patience sorting".

    tails[k] = the smallest possible last value of an increasing subsequence of
    length k + 1 seen so far. tails is always sorted, so for each x we binary
    search for the first tail >= x and replace it (x is a better, smaller end
    for that length); if none exists, x extends the longest subsequence.
    """
    tails: list[int] = []
    for x in nums:
        i = bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def lis_sequence(nums: Sequence[int]) -> list[int]:
    """One actual longest strictly increasing subsequence, still O(n log n).

    Alongside tails we remember, for every element, which element comes before
    it in the best subsequence ending there, then walk those links backwards.
    """
    tails: list[int] = []  # values, as in lis_length
    tails_idx: list[int] = []  # index in nums of each tail
    prev = [-1] * len(nums)
    for i, x in enumerate(nums):
        k = bisect_left(tails, x)
        if k > 0:
            prev[i] = tails_idx[k - 1]
        if k == len(tails):
            tails.append(x)
            tails_idx.append(i)
        else:
            tails[k] = x
            tails_idx[k] = i
    out = []
    i = tails_idx[-1] if tails_idx else -1
    while i != -1:
        out.append(nums[i])
        i = prev[i]
    return out[::-1]


def _lcs_table(a: Sequence, b: Sequence) -> list[list[int]]:
    """L[i][j] = LCS length of a[:i] and b[:j]."""
    L = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1  # matching last items extend the LCS
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])  # drop one item from a or b
    return L


def lcs_length(a: Sequence, b: Sequence) -> int:
    return _lcs_table(a, b)[len(a)][len(b)]


def lcs_string(a: str, b: str) -> str:
    """One longest common subsequence, rebuilt by walking the table from the corner."""
    L = _lcs_table(a, b)
    i, j, out = len(a), len(b), []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(a[i - 1])
            i, j = i - 1, j - 1
        elif L[i - 1][j] >= L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(out))


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance: fewest insertions, deletions and substitutions
    turning a into b. O(len(a) * len(b)) time, O(len(b)) space.

    D[i][j] = distance between a[:i] and b[:j]
            = D[i-1][j-1]                    if a[i-1] == b[j-1]
            = 1 + min(D[i-1][j],             delete a[i-1]
                      D[i][j-1],             insert b[j-1]
                      D[i-1][j-1])           substitute
    """
    prev = list(range(len(b) + 1))  # D[0][j] = j insertions
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)  # D[i][0] = i deletions
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[len(b)]


def edit_operations(a: str, b: str) -> list[str]:
    """A shortest list of human-readable edits turning a into b."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = a[i - 1] == b[j - 1]
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (not same))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and D[i][j] == D[i - 1][j - 1]:
            i, j = i - 1, j - 1  # characters match: no edit
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            ops.append(f"substitute {a[i - 1]}->{b[j - 1]} at {i - 1}")
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(f"delete {a[i - 1]} at {i - 1}")
            i -= 1
        else:
            ops.append(f"insert {b[j - 1]} at {i}")
            j -= 1
    return ops[::-1]


def longest_palindromic_subsequence(s: str) -> int:
    """A palindrome reads the same reversed, so the answer is LCS(s, reversed s)."""
    return lcs_length(s, s[::-1])
