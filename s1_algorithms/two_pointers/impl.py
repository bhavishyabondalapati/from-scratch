"""Two pointers: move two indices through a sequence instead of nesting two loops."""

from __future__ import annotations


def is_palindrome(s: str) -> bool:
    """True if s reads the same both ways, ignoring case and non-alphanumerics."""
    i, j = 0, len(s) - 1
    while i < j:
        if not s[i].isalnum():
            i += 1
        elif not s[j].isalnum():
            j -= 1
        elif s[i].lower() != s[j].lower():
            return False
        else:
            i, j = i + 1, j - 1
    return True


def two_sum_sorted(nums: list[int], target: int) -> tuple[int, int] | None:
    """Pair of indices in a SORTED list summing to target. O(n) time, O(1) space.

    If the sum is too small, only moving the left pointer right can increase it;
    if too big, only moving the right pointer left can decrease it.
    """
    i, j = 0, len(nums) - 1
    while i < j:
        s = nums[i] + nums[j]
        if s == target:
            return i, j
        if s < target:
            i += 1
        else:
            j -= 1
    return None


def three_sum(nums: list[int]) -> list[tuple[int, int, int]]:
    """All unique triplets (a <= b <= c) with a + b + c == 0. O(n^2)."""
    nums = sorted(nums)
    out: list[tuple[int, int, int]] = []
    for k in range(len(nums) - 2):
        if k > 0 and nums[k] == nums[k - 1]:
            continue  # same first element as last time -> would repeat triplets
        i, j = k + 1, len(nums) - 1
        while i < j:
            s = nums[k] + nums[i] + nums[j]
            if s < 0:
                i += 1
            elif s > 0:
                j -= 1
            else:
                out.append((nums[k], nums[i], nums[j]))
                i, j = i + 1, j - 1
                while i < j and nums[i] == nums[i - 1]:
                    i += 1  # skip duplicate second elements
    return out


def max_water_container(heights: list[int]) -> int:
    """Largest area min(h[i], h[j]) * (j - i). O(n).

    Start as wide as possible. The shorter wall limits the area, and moving the
    taller wall inward can never help, so always move the shorter one.
    """
    i, j, best = 0, len(heights) - 1, 0
    while i < j:
        best = max(best, min(heights[i], heights[j]) * (j - i))
        if heights[i] < heights[j]:
            i += 1
        else:
            j -= 1
    return best


def remove_duplicates_sorted(nums: list[int]) -> int:
    """Deduplicate a sorted list in place; return the new length (slow/fast pointers)."""
    if not nums:
        return 0
    write = 1  # next position to write a new distinct value
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1
    return write


def sort_colors(nums: list[int]) -> None:
    """Dutch national flag: sort a list of 0/1/2 in place in one pass.

    Invariant: nums[:lo] are 0s, nums[lo:mid] are 1s, nums[hi+1:] are 2s,
    nums[mid:hi+1] is still unknown.
    """
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo, mid = lo + 1, mid + 1
        elif nums[mid] == 1:
            mid += 1
        else:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1  # do NOT advance mid: the swapped-in value is unexamined


def trap_rain_water(heights: list[int]) -> int:
    """Water trapped between bars. O(n) time, O(1) space.

    Water above bar i = min(max to its left, max to its right) - height[i].
    The side with the smaller running max is the binding one, so process it.
    """
    i, j = 0, len(heights) - 1
    left_max = right_max = total = 0
    while i < j:
        if heights[i] < heights[j]:
            left_max = max(left_max, heights[i])
            total += left_max - heights[i]
            i += 1
        else:
            right_max = max(right_max, heights[j])
            total += right_max - heights[j]
            j -= 1
    return total
