import random
import statistics

import numpy as np
import pytest

from .impl import k_closest_points, kth_largest, median, median_of_medians_select, quickselect


@pytest.mark.parametrize("select", [quickselect, median_of_medians_select])
def test_select_matches_sorted_for_every_k(select):
    rnd = random.Random(0)
    for _ in range(200):
        nums = [rnd.randint(-20, 20) for _ in range(rnd.randint(1, 40))]
        original, ordered = list(nums), sorted(nums)
        for k in range(len(nums)):
            assert select(nums, k) == ordered[k]
        assert nums == original
        with pytest.raises(IndexError):
            select(nums, len(nums))


def test_matches_numpy_partition_on_large_input():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 10**6, size=50_000)
    nums = arr.tolist()
    for k in (0, 1, 12_345, 25_000, 49_999):
        expected = int(np.partition(arr, k)[k])
        assert quickselect(nums, k) == expected
        assert median_of_medians_select(nums, k) == expected


def test_adversarial_inputs_stay_fast():
    n = 100_000
    assert quickselect([7] * n, n // 2) == 7  # all equal
    assert quickselect(list(range(n)), n // 3) == n // 3  # sorted
    assert median_of_medians_select(list(range(n, 0, -1)), 10) == 11


def test_kth_largest_and_median():
    rnd = random.Random(2)
    for _ in range(200):
        nums = [rnd.randint(-30, 30) for _ in range(rnd.randint(1, 25))]
        k = rnd.randint(1, len(nums))
        assert kth_largest(nums, k) == sorted(nums, reverse=True)[k - 1]
        assert median(nums) == statistics.median(nums)
    with pytest.raises(ValueError):
        median([])


def test_k_closest_points():
    rnd = random.Random(3)
    for _ in range(200):
        points = [(rnd.randint(-5, 5), rnd.randint(-5, 5)) for _ in range(rnd.randint(0, 15))]
        k = rnd.randint(0, 17)
        got = k_closest_points(points, k)
        dist = lambda p: p[0] ** 2 + p[1] ** 2  # noqa: E731
        assert len(got) == min(k, len(points))
        # Same multiset of distances as the k smallest (ties may be broken differently).
        assert sorted(map(dist, got)) == sorted(map(dist, points))[: len(got)]
        remaining = list(points)
        for p in got:
            remaining.remove(p)  # every returned point is a distinct input point
