"""Binary search: halve the search space using a yes/no question that flips once."""

from __future__ import annotations

import math
from collections.abc import Callable


def binary_search(nums: list[int], target: int) -> int:
    """Index of target in sorted nums, or -1."""
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def lower_bound(nums: list[int], target: int) -> int:
    """First index i with nums[i] >= target (len(nums) if none). Same as bisect_left."""
    lo, hi = 0, len(nums)  # answer lies in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1  # mid is too small, answer is strictly right of it
        else:
            hi = mid  # mid might be the answer, keep it
    return lo


def upper_bound(nums: list[int], target: int) -> int:
    """First index i with nums[i] > target. Same as bisect_right."""
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def first_true(lo: int, hi: int, pred: Callable[[int], bool]) -> int:
    """Smallest x in [lo, hi] with pred(x) True, assuming pred is False...False True...True.

    Returns hi + 1 if pred is never True. This is the general template: every
    binary search is "find the boundary of a monotonic predicate".
    """
    hi += 1  # hi is now "one past the end", the answer if nothing is True
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def search_rotated(nums: list[int], target: int) -> int:
    """Search a sorted array of distinct values that was rotated at some pivot."""
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:  # left half [lo, mid] is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:  # right half [mid, hi] is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def find_min_rotated(nums: list[int]) -> int:
    """Minimum of a rotated sorted array of distinct values."""
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > nums[hi]:
            lo = mid + 1  # the drop (minimum) is to the right of mid
        else:
            hi = mid
    return nums[lo]


def integer_sqrt(n: int) -> int:
    """Largest r with r*r <= n, by searching the answer space [0, n]."""
    return first_true(0, n, lambda r: r * r > n) - 1


# --- Binary search on the answer space -------------------------------------
# Pattern: "minimise X such that a feasibility check passes", where feasibility
# is monotonic in X (if speed 5 works, speed 6 works too).


def min_eating_speed(piles: list[int], hours: int) -> int:
    """Smallest speed k (items/hour) to finish all piles within `hours`."""

    def can_finish(k: int) -> bool:
        return sum(math.ceil(p / k) for p in piles) <= hours

    return first_true(1, max(piles), can_finish)


def split_array_min_largest_sum(nums: list[int], k: int) -> int:
    """Split non-negative nums into k contiguous parts minimising the largest part sum."""

    def can_split(limit: int) -> bool:
        parts, running = 1, 0
        for x in nums:
            if running + x > limit:  # start a new part
                parts, running = parts + 1, 0
            running += x
        return parts <= k

    # No part can be smaller than the largest element or bigger than the total.
    return first_true(max(nums), sum(nums), can_split)
