import random
from itertools import accumulate

from .impl import FenwickTree, RangeUpdateFenwick, count_inversions


def test_fenwick_matches_list_under_random_operations():
    rnd = random.Random(0)
    for n in [1, 2, 3, 8, 13, 31]:
        values = [rnd.randint(-9, 9) for _ in range(n)]
        tree = FenwickTree(values)
        for _ in range(500):
            if rnd.random() < 0.4:
                i, d = rnd.randrange(n), rnd.randint(-9, 9)
                values[i] += d
                tree.add(i, d)
            lo = rnd.randint(0, n)
            hi = rnd.randint(lo, n)
            assert tree.range_sum(lo, hi) == sum(values[lo:hi])
            assert tree.prefix_sum(hi) == sum(values[:hi])


def test_build_from_size_equals_all_zeros():
    tree = FenwickTree(5)
    assert tree.prefix_sum(5) == 0
    tree.add(4, 7)
    assert tree.range_sum(4, 5) == 7 and tree.range_sum(0, 4) == 0


def test_find_kth_matches_linear_scan():
    rnd = random.Random(1)
    for _ in range(300):
        n = rnd.randint(1, 20)
        values = [rnd.randint(0, 4) for _ in range(n)]
        tree = FenwickTree(values)
        prefix = list(accumulate(values))
        for k in range(1, sum(values) + 3):
            expected = next((i for i, p in enumerate(prefix) if p >= k), n)
            assert tree.find_kth(k) == expected


def test_range_update_point_query():
    rnd = random.Random(2)
    n = 20
    ref, tree = [0] * n, RangeUpdateFenwick(n)
    for _ in range(500):
        lo = rnd.randint(0, n)
        hi = rnd.randint(lo, n)
        d = rnd.randint(-5, 5)
        tree.range_add(lo, hi, d)
        for i in range(lo, hi):
            ref[i] += d
        i = rnd.randrange(n)
        assert tree.point_query(i) == ref[i]


def test_count_inversions_against_brute_force():
    rnd = random.Random(3)
    assert count_inversions([]) == 0 and count_inversions([3, 2, 1]) == 3
    for _ in range(300):
        nums = [rnd.randint(-5, 5) for _ in range(rnd.randint(0, 15))]
        n = len(nums)
        expected = sum(nums[i] > nums[j] for i in range(n) for j in range(i + 1, n))
        assert count_inversions(nums) == expected
