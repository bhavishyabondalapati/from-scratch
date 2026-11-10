import random

from .impl import (
    PrefixSum,
    PrefixSum2D,
    apply_range_updates,
    longest_subarray_with_sum,
    pivot_index,
    subarray_sum_equals_k,
)


def test_prefix_sum_matches_builtin_sum():
    rnd = random.Random(0)
    for _ in range(200):
        nums = [rnd.randint(-9, 9) for _ in range(rnd.randint(0, 20))]
        ps = PrefixSum(nums)
        for _ in range(20):
            lo = rnd.randint(0, len(nums))
            hi = rnd.randint(lo, len(nums))
            assert ps.range_sum(lo, hi) == sum(nums[lo:hi])


def test_prefix_sum_2d_matches_brute_force():
    rnd = random.Random(1)
    for _ in range(100):
        rows, cols = rnd.randint(1, 6), rnd.randint(1, 6)
        grid = [[rnd.randint(-5, 5) for _ in range(cols)] for _ in range(rows)]
        ps = PrefixSum2D(grid)
        for _ in range(20):
            r1 = rnd.randint(0, rows)
            r2 = rnd.randint(r1, rows)
            c1 = rnd.randint(0, cols)
            c2 = rnd.randint(c1, cols)
            assert ps.region_sum(r1, c1, r2, c2) == sum(sum(row[c1:c2]) for row in grid[r1:r2])


def test_subarray_sum_equals_k():
    rnd = random.Random(2)
    for _ in range(300):
        nums = [rnd.randint(-4, 4) for _ in range(rnd.randint(0, 12))]
        k = rnd.randint(-5, 5)
        n = len(nums)
        expected = sum(sum(nums[i:j]) == k for i in range(n) for j in range(i + 1, n + 1))
        assert subarray_sum_equals_k(nums, k) == expected


def test_apply_range_updates():
    rnd = random.Random(3)
    for _ in range(200):
        n = rnd.randint(1, 15)
        updates, expected = [], [0] * n
        for _ in range(rnd.randint(0, 10)):
            lo = rnd.randint(0, n)
            hi = rnd.randint(lo, n)
            v = rnd.randint(-5, 5)
            updates.append((lo, hi, v))
            for i in range(lo, hi):
                expected[i] += v
        assert apply_range_updates(n, updates) == expected


def test_pivot_index():
    assert pivot_index([1, 7, 3, 6, 5, 6]) == 3
    rnd = random.Random(4)
    for _ in range(300):
        nums = [rnd.randint(-3, 3) for _ in range(rnd.randint(0, 10))]
        expected = next((i for i in range(len(nums)) if sum(nums[:i]) == sum(nums[i + 1 :])), -1)
        assert pivot_index(nums) == expected


def test_longest_subarray_with_sum():
    rnd = random.Random(5)
    for _ in range(300):
        nums = [rnd.randint(-3, 3) for _ in range(rnd.randint(0, 12))]
        k = rnd.randint(-4, 4)
        n = len(nums)
        expected = max(
            (j - i for i in range(n) for j in range(i + 1, n + 1) if sum(nums[i:j]) == k),
            default=0,
        )
        assert longest_subarray_with_sum(nums, k) == expected
