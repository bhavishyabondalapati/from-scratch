import bisect
import math
import random
from itertools import combinations

from .impl import (
    binary_search,
    find_min_rotated,
    first_true,
    integer_sqrt,
    lower_bound,
    min_eating_speed,
    search_rotated,
    split_array_min_largest_sum,
    upper_bound,
)


def test_bounds_match_bisect_module():
    rnd = random.Random(0)
    for _ in range(500):
        nums = sorted(rnd.randint(0, 20) for _ in range(rnd.randint(0, 20)))
        t = rnd.randint(-2, 22)
        assert lower_bound(nums, t) == bisect.bisect_left(nums, t)
        assert upper_bound(nums, t) == bisect.bisect_right(nums, t)
        i = binary_search(nums, t)
        assert (i == -1) == (t not in nums)
        if i != -1:
            assert nums[i] == t


def test_first_true_finds_boundary():
    for boundary in range(0, 12):
        assert first_true(0, 10, lambda x, b=boundary: x >= b) == boundary


def test_rotated_search_and_min():
    rnd = random.Random(1)
    for _ in range(500):
        base = sorted(rnd.sample(range(50), rnd.randint(1, 15)))
        r = rnd.randint(0, len(base) - 1)
        nums = base[r:] + base[:r]
        assert find_min_rotated(nums) == min(nums)
        t = rnd.randint(0, 49)
        expected = nums.index(t) if t in nums else -1
        assert search_rotated(nums, t) == expected


def test_integer_sqrt():
    for n in list(range(0, 300)) + [10**12, 10**12 + 1, 2**61 - 1]:
        assert integer_sqrt(n) == math.isqrt(n)


def test_min_eating_speed_against_linear_scan():
    rnd = random.Random(2)
    for _ in range(200):
        piles = [rnd.randint(1, 30) for _ in range(rnd.randint(1, 8))]
        hours = rnd.randint(len(piles), 40)
        expected = next(
            k for k in range(1, max(piles) + 1) if sum(math.ceil(p / k) for p in piles) <= hours
        )
        assert min_eating_speed(piles, hours) == expected


def test_split_array_against_brute_force():
    rnd = random.Random(3)
    for _ in range(200):
        nums = [rnd.randint(0, 9) for _ in range(rnd.randint(1, 8))]
        k = rnd.randint(1, len(nums))
        best = math.inf
        for cuts in combinations(range(1, len(nums)), k - 1):  # every way to cut into k parts
            edges = [0, *cuts, len(nums)]
            best = min(best, max(sum(nums[a:b]) for a, b in zip(edges, edges[1:])))
        assert split_array_min_largest_sum(nums, k) == best
