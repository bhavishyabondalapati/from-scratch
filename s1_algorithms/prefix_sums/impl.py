"""Prefix sums: precompute running totals so any range sum is one subtraction."""

from __future__ import annotations


class PrefixSum:
    """Immutable 1D range-sum queries in O(1) after O(n) setup.

    prefix[i] = sum of the first i elements, so prefix[0] = 0 and
    sum(nums[lo:hi]) = prefix[hi] - prefix[lo].
    """

    def __init__(self, nums: list[int]) -> None:
        self.prefix = [0]
        for x in nums:
            self.prefix.append(self.prefix[-1] + x)

    def range_sum(self, lo: int, hi: int) -> int:
        """Sum of nums[lo:hi] (hi exclusive)."""
        return self.prefix[hi] - self.prefix[lo]


class PrefixSum2D:
    """2D range sums using inclusion-exclusion.

    P[r][c] = sum of the rectangle from (0, 0) up to but excluding row r, column c.
    """

    def __init__(self, grid: list[list[int]]) -> None:
        rows, cols = len(grid), len(grid[0]) if grid else 0
        self.P = [[0] * (cols + 1) for _ in range(rows + 1)]
        for r in range(rows):
            for c in range(cols):
                self.P[r + 1][c + 1] = (
                    grid[r][c] + self.P[r][c + 1] + self.P[r + 1][c] - self.P[r][c]
                )

    def region_sum(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Sum of grid[r1:r2][c1:c2] (r2, c2 exclusive)."""
        P = self.P
        return P[r2][c2] - P[r1][c2] - P[r2][c1] + P[r1][c1]


def subarray_sum_equals_k(nums: list[int], k: int) -> int:
    """Number of contiguous subarrays summing to k (negatives allowed). O(n).

    sum(nums[i:j]) == k  <=>  prefix[j] - prefix[i] == k  <=>  prefix[i] == prefix[j] - k.
    So for each j, count how many earlier prefixes equal prefix[j] - k.
    """
    seen = {0: 1}  # prefix value -> how many times seen (empty prefix counts once)
    running = count = 0
    for x in nums:
        running += x
        count += seen.get(running - k, 0)
        seen[running] = seen.get(running, 0) + 1
    return count


def apply_range_updates(n: int, updates: list[tuple[int, int, int]]) -> list[int]:
    """Difference array: apply many "add v to arr[lo:hi]" updates in O(n + updates).

    Mark +v where a range starts and -v just after it ends; a prefix sum over the
    marks then rebuilds the final array.
    """
    diff = [0] * (n + 1)
    for lo, hi, v in updates:
        diff[lo] += v
        diff[hi] -= v
    out, running = [], 0
    for i in range(n):
        running += diff[i]
        out.append(running)
    return out


def pivot_index(nums: list[int]) -> int:
    """Leftmost i where sum(nums[:i]) == sum(nums[i+1:]), or -1."""
    total, left = sum(nums), 0
    for i, x in enumerate(nums):
        if left == total - left - x:
            return i
        left += x
    return -1


def longest_subarray_with_sum(nums: list[int], k: int) -> int:
    """Length of the longest subarray summing to k (negatives allowed). O(n)."""
    first_index = {0: -1}  # prefix value -> earliest index where it occurred
    running = best = 0
    for j, x in enumerate(nums):
        running += x
        if running - k in first_index:
            best = max(best, j - first_index[running - k])
        first_index.setdefault(running, j)  # keep the EARLIEST index only
    return best
