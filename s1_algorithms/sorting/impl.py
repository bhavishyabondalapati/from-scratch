"""Sorting algorithms: quick, merge, heap, counting, radix (plus insertion sort).

Every function returns a new sorted list and leaves the input unchanged.
"""

from __future__ import annotations

import random


def insertion_sort(nums: list[int]) -> list[int]:
    """Grow a sorted prefix by inserting one element at a time. O(n^2), stable.

    Fast on tiny or nearly sorted inputs, which is why real sorts use it as a base case.
    """
    a = list(nums)
    for i in range(1, len(a)):
        x, j = a[i], i - 1
        while j >= 0 and a[j] > x:  # shift bigger elements one slot right
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x
    return a


def merge_sort(nums: list[int]) -> list[int]:
    """Split in half, sort each half, merge. O(n log n) always, stable, O(n) extra."""
    if len(nums) <= 1:
        return list(nums)
    mid = len(nums) // 2
    left, right = merge_sort(nums[:mid]), merge_sort(nums[mid:])
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:  # "<=" takes from the left on ties -> stable
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def quick_sort(nums: list[int], seed: int | None = 0) -> list[int]:
    """In-place quicksort with a random pivot and 3-way partitioning.

    Average O(n log n), worst case O(n^2) (made very unlikely by the random
    pivot). Not stable. 3-way partitioning groups all copies of the pivot in
    the middle, so inputs with many duplicates stay fast.
    """
    a = list(nums)
    rnd = random.Random(seed)

    def sort(lo: int, hi: int) -> None:
        while lo < hi:
            pivot = a[rnd.randint(lo, hi)]
            # Dutch national flag: a[lo:lt] < pivot, a[lt:i] == pivot, a[gt+1:hi+1] > pivot
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
            # Recurse into the smaller side, loop on the larger: stack depth O(log n).
            if lt - lo < hi - gt:
                sort(lo, lt - 1)
                lo = gt + 1
            else:
                sort(gt + 1, hi)
                hi = lt - 1

    sort(0, len(a) - 1)
    return a


def heap_sort(nums: list[int]) -> list[int]:
    """Build a max-heap, then repeatedly move the max to the end. O(n log n), in place, not stable."""
    a = list(nums)
    n = len(a)

    def sift_down(i: int, size: int) -> None:
        while True:
            largest = i
            for child in (2 * i + 1, 2 * i + 2):
                if child < size and a[child] > a[largest]:
                    largest = child
            if largest == i:
                return
            a[i], a[largest] = a[largest], a[i]
            i = largest

    for i in range(n // 2 - 1, -1, -1):  # heapify in O(n)
        sift_down(i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]  # the max goes to its final position
        sift_down(0, end)  # restore the heap on the shrunken prefix
    return a


def counting_sort(nums: list[int]) -> list[int]:
    """Sort integers by counting occurrences. O(n + k) where k = max - min + 1. Stable.

    No comparisons at all, so the O(n log n) lower bound does not apply.
    """
    if not nums:
        return []
    lo, hi = min(nums), max(nums)
    counts = [0] * (hi - lo + 1)
    for x in nums:
        counts[x - lo] += 1
    # Prefix sums turn counts into "index just past the last slot for this value".
    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]
    out = [0] * len(nums)
    for x in reversed(nums):  # reversed scan keeps equal elements in original order
        counts[x - lo] -= 1
        out[counts[x - lo]] = x
    return out


def radix_sort(nums: list[int], base: int = 10) -> list[int]:
    """LSD radix sort: stable-sort by each digit, least significant first.

    O(d * (n + base)) for d digits. Correct because each pass is STABLE: when
    sorting by a higher digit, ties keep the order set by the lower digits.
    Negative numbers are sorted by magnitude separately and reversed.
    """

    def sort_non_negative(a: list[int]) -> list[int]:
        if not a:
            return a
        place, biggest = 1, max(a)
        while place <= biggest:
            buckets: list[list[int]] = [[] for _ in range(base)]
            for x in a:
                buckets[(x // place) % base].append(x)  # appending preserves order -> stable
            a = [x for bucket in buckets for x in bucket]
            place *= base
        return a

    negatives = sort_non_negative([-x for x in nums if x < 0])
    return [-x for x in reversed(negatives)] + sort_non_negative([x for x in nums if x >= 0])
