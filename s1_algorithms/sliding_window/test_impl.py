import random
from collections import Counter

import pytest

from .impl import (
    find_anagram_starts,
    longest_repeating_replacement,
    longest_unique_substring,
    max_sum_subarray_k,
    min_subarray_len,
    min_window_substring,
)


def substrings(s):
    return [(i, j) for i in range(len(s)) for j in range(i, len(s))]


def test_max_sum_subarray_k():
    rnd = random.Random(0)
    for _ in range(300):
        nums = [rnd.randint(-9, 9) for _ in range(rnd.randint(1, 15))]
        k = rnd.randint(1, len(nums))
        assert max_sum_subarray_k(nums, k) == max(
            sum(nums[i : i + k]) for i in range(len(nums) - k + 1)
        )
    with pytest.raises(ValueError):
        max_sum_subarray_k([1, 2], 3)


def test_longest_unique_substring():
    rnd = random.Random(1)
    for _ in range(300):
        s = "".join(rnd.choice("abcd") for _ in range(rnd.randint(0, 12)))
        expected = max(
            (j - i + 1 for i, j in substrings(s) if len(set(s[i : j + 1])) == j - i + 1), default=0
        )
        assert longest_unique_substring(s) == expected


def test_min_subarray_len():
    rnd = random.Random(2)
    for _ in range(300):
        nums = [rnd.randint(1, 9) for _ in range(rnd.randint(0, 12))]
        target = rnd.randint(1, 40)
        expected = min(
            (j - i + 1 for i, j in substrings(nums) if sum(nums[i : j + 1]) >= target), default=0
        )
        assert min_subarray_len(target, nums) == expected


def test_longest_repeating_replacement():
    rnd = random.Random(3)
    for _ in range(300):
        s = "".join(rnd.choice("ABC") for _ in range(rnd.randint(0, 12)))
        k = rnd.randint(0, 3)
        expected = max(
            (
                j - i + 1
                for i, j in substrings(s)
                if (j - i + 1) - max(Counter(s[i : j + 1]).values()) <= k
            ),
            default=0,
        )
        assert longest_repeating_replacement(s, k) == expected


def test_min_window_substring():
    assert min_window_substring("ADOBECODEBANC", "ABC") == "BANC"
    rnd = random.Random(4)
    for _ in range(300):
        s = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 12)))
        t = "".join(rnd.choice("abc") for _ in range(rnd.randint(1, 4)))
        valid = [s[i : j + 1] for i, j in substrings(s) if not (Counter(t) - Counter(s[i : j + 1]))]
        got = min_window_substring(s, t)
        if not valid:
            assert got == ""
        else:
            assert got in valid and len(got) == min(map(len, valid))


def test_find_anagram_starts():
    rnd = random.Random(5)
    for _ in range(300):
        s = "".join(rnd.choice("ab") for _ in range(rnd.randint(0, 12)))
        p = "".join(rnd.choice("ab") for _ in range(rnd.randint(1, 4)))
        expected = [
            i for i in range(len(s) - len(p) + 1) if sorted(s[i : i + len(p)]) == sorted(p)
        ]
        assert find_anagram_starts(s, p) == expected
