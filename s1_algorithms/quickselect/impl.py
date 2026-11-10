"""Quickselect: find the k-th smallest element without fully sorting."""

from __future__ import annotations

import random


def quickselect(nums: list[int], k: int, seed: int | None = 0) -> int:
    """The k-th smallest element (k = 0 is the minimum). Average O(n), worst O(n^2).

    Partition around a random pivot, exactly like quicksort, but then continue
    into only the ONE side that contains position k. The work shrinks
    geometrically: n + n/2 + n/4 + ... = 2n on average.
    """
    if not 0 <= k < len(nums):
        raise IndexError("k out of range")
    a = list(nums)
    rnd = random.Random(seed)
    lo, hi = 0, len(a) - 1
    while True:
        pivot = a[rnd.randint(lo, hi)]
        # 3-way partition: a[lo:lt] < pivot, a[lt:gt+1] == pivot, a[gt+1:hi+1] > pivot
        lt, i, gt = lo, lo, hi
        while i <= gt:
            if a[i] < pivot:
                a[lt], a[i] = a[i], a[lt]
                lt, i = lt + 1, i + 1
            elif a[i] > pivot:
                a[i], a[gt] = a[gt], a[i]
                gt -= 1
            else:
                i += 1
        if k < lt:
            hi = lt - 1  # answer is among the smaller elements
        elif k > gt:
            lo = gt + 1  # answer is among the larger elements
        else:
            return pivot  # k falls inside the block equal to the pivot


def kth_largest(nums: list[int], k: int) -> int:
    """The k-th largest element (k = 1 is the maximum)."""
    return quickselect(nums, len(nums) - k)


def median(nums: list[int]) -> float:
    n = len(nums)
    if n == 0:
        raise ValueError("median of empty list")
    if n % 2:
        return float(quickselect(nums, n // 2))
    return (quickselect(nums, n // 2 - 1) + quickselect(nums, n // 2)) / 2


def k_closest_points(points: list[tuple[float, float]], k: int) -> list[tuple[float, float]]:
    """The k points nearest the origin (any order). Average O(n)."""
    if k <= 0:
        return []
    if k >= len(points):
        return list(points)
    sq_dist = [x * x + y * y for x, y in points]
    threshold = quickselect(sq_dist, k - 1)  # distance of the k-th closest point
    closer = [p for p, d in zip(points, sq_dist) if d < threshold]
    ties = [p for p, d in zip(points, sq_dist) if d == threshold]
    return closer + ties[: k - len(closer)]


def median_of_medians_select(nums: list[int], k: int) -> int:
    """Deterministic selection with a guaranteed O(n) worst case.

    Choose the pivot as the median of the medians of groups of 5. At least
    ~30% of the elements are guaranteed smaller and ~30% larger than it, so
    each round discards a constant fraction: T(n) = T(n/5) + T(7n/10) + O(n) = O(n).
    """
    if not 0 <= k < len(nums):
        raise IndexError("k out of range")
    a = list(nums)
    while True:
        if len(a) <= 5:
            return sorted(a)[k]
        medians = [sorted(a[i : i + 5])[len(a[i : i + 5]) // 2] for i in range(0, len(a), 5)]
        pivot = median_of_medians_select(medians, len(medians) // 2)
        smaller = [x for x in a if x < pivot]
        equal_count = a.count(pivot)
        if k < len(smaller):
            a = smaller
        elif k < len(smaller) + equal_count:
            return pivot
        else:
            k -= len(smaller) + equal_count
            a = [x for x in a if x > pivot]
