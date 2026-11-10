import random
from itertools import combinations

from .impl import (
    is_palindrome,
    max_water_container,
    remove_duplicates_sorted,
    sort_colors,
    three_sum,
    trap_rain_water,
    two_sum_sorted,
)


def test_is_palindrome():
    assert is_palindrome("A man, a plan, a canal: Panama")
    assert not is_palindrome("race a car")
    assert is_palindrome("") and is_palindrome(".,")
    rnd = random.Random(0)
    for _ in range(300):
        s = "".join(rnd.choice("ab ,A") for _ in range(rnd.randint(0, 8)))
        clean = [c.lower() for c in s if c.isalnum()]
        assert is_palindrome(s) == (clean == clean[::-1])


def test_two_sum_sorted_against_brute_force():
    rnd = random.Random(1)
    for _ in range(300):
        nums = sorted(rnd.randint(-10, 10) for _ in range(rnd.randint(0, 10)))
        target = rnd.randint(-20, 20)
        exists = any(a + b == target for a, b in combinations(nums, 2))
        got = two_sum_sorted(nums, target)
        assert (got is not None) == exists
        if got:
            assert got[0] < got[1] and nums[got[0]] + nums[got[1]] == target


def test_three_sum_against_brute_force():
    rnd = random.Random(2)
    for _ in range(200):
        nums = [rnd.randint(-6, 6) for _ in range(rnd.randint(0, 12))]
        expected = {tuple(sorted(c)) for c in combinations(nums, 3) if sum(c) == 0}
        got = three_sum(nums)
        assert len(got) == len(set(got))  # no duplicates
        assert set(got) == expected


def test_max_water_container_against_brute_force():
    rnd = random.Random(3)
    for _ in range(300):
        h = [rnd.randint(0, 20) for _ in range(rnd.randint(2, 15))]
        expected = max(min(h[i], h[j]) * (j - i) for i, j in combinations(range(len(h)), 2))
        assert max_water_container(h) == expected


def test_remove_duplicates_sorted():
    rnd = random.Random(4)
    for _ in range(300):
        nums = sorted(rnd.randint(0, 6) for _ in range(rnd.randint(0, 15)))
        expected = sorted(set(nums))
        k = remove_duplicates_sorted(nums)
        assert nums[:k] == expected


def test_sort_colors():
    rnd = random.Random(5)
    for _ in range(300):
        nums = [rnd.randint(0, 2) for _ in range(rnd.randint(0, 20))]
        expected = sorted(nums)
        sort_colors(nums)
        assert nums == expected


def test_trap_rain_water_against_definition():
    rnd = random.Random(6)
    assert trap_rain_water([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6
    for _ in range(300):
        h = [rnd.randint(0, 9) for _ in range(rnd.randint(0, 15))]
        expected = sum(min(max(h[: i + 1]), max(h[i:])) - h[i] for i in range(len(h)))
        assert trap_rain_water(h) == expected
