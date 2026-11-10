"""Sliding window: maintain a contiguous range [left, right] and update it incrementally."""

from __future__ import annotations

from collections import Counter


def max_sum_subarray_k(nums: list[int], k: int) -> int:
    """Largest sum of any k consecutive elements (fixed-size window). O(n)."""
    if k <= 0 or k > len(nums):
        raise ValueError("k must be in 1..len(nums)")
    window = sum(nums[:k])
    best = window
    for right in range(k, len(nums)):
        window += nums[right] - nums[right - k]  # add the new item, drop the oldest
        best = max(best, window)
    return best


def longest_unique_substring(s: str) -> int:
    """Length of the longest substring with no repeated character. O(n)."""
    last_seen: dict[str, int] = {}
    left = best = 0
    for right, ch in enumerate(s):
        # If ch is already inside the window, jump left just past its old position.
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best


def min_subarray_len(target: int, nums: list[int]) -> int:
    """Shortest subarray of POSITIVE numbers with sum >= target (0 if none). O(n)."""
    left = total = 0
    best = len(nums) + 1
    for right, x in enumerate(nums):
        total += x
        while total >= target:  # window is valid: record it, then try to shrink
            best = min(best, right - left + 1)
            total -= nums[left]
            left += 1
    return 0 if best > len(nums) else best


def longest_repeating_replacement(s: str, k: int) -> int:
    """Longest substring that becomes one repeated letter after <= k replacements.

    A window is valid when (window length - count of its most common letter) <= k.
    """
    counts: Counter[str] = Counter()
    left = best = max_count = 0
    for right, ch in enumerate(s):
        counts[ch] += 1
        max_count = max(max_count, counts[ch])
        # max_count may be stale (too large) after shrinking, but that is fine:
        # a stale value never makes the answer larger than a window we truly saw.
        if (right - left + 1) - max_count > k:
            counts[s[left]] -= 1
            left += 1
        best = max(best, right - left + 1)
    return best


def min_window_substring(s: str, t: str) -> str:
    """Shortest substring of s containing every character of t (with multiplicity)."""
    if not t:
        return ""
    need = Counter(t)
    missing = len(t)  # how many required characters the window still lacks
    left = 0
    best: tuple[int, int] | None = None
    for right, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1
        need[ch] -= 1  # negative means the window has a surplus of ch
        while missing == 0:
            if best is None or right - left < best[1] - best[0]:
                best = (left, right)
            need[s[left]] += 1
            if need[s[left]] > 0:  # we just dropped a character we needed
                missing += 1
            left += 1
    return "" if best is None else s[best[0] : best[1] + 1]


def find_anagram_starts(s: str, p: str) -> list[int]:
    """Start indices of substrings of s that are anagrams of p. O(len(s))."""
    n, m = len(s), len(p)
    if m == 0 or m > n:
        return []
    need, window = Counter(p), Counter(s[:m])
    out = [0] if window == need else []
    for right in range(m, n):
        window[s[right]] += 1
        old = s[right - m]
        window[old] -= 1
        if window[old] == 0:
            del window[old]  # keep the Counter comparable with ==
        if window == need:
            out.append(right - m + 1)
    return out
