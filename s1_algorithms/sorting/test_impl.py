import random
import sys

import pytest

from .impl import counting_sort, heap_sort, insertion_sort, merge_sort, quick_sort, radix_sort

ALL = [insertion_sort, merge_sort, quick_sort, heap_sort, counting_sort, radix_sort]


@pytest.mark.parametrize("sort", ALL, ids=lambda f: f.__name__)
def test_matches_builtin_sorted_on_random_inputs(sort):
    rnd = random.Random(0)
    for _ in range(300):
        nums = [rnd.randint(-50, 50) for _ in range(rnd.randint(0, 40))]
        original = list(nums)
        assert sort(nums) == sorted(nums)
        assert nums == original  # input is not modified


@pytest.mark.parametrize("sort", ALL, ids=lambda f: f.__name__)
def test_edge_cases(sort):
    for nums in ([], [7], [2, 1], [3, 3, 3, 3], list(range(50)), list(range(50, 0, -1))):
        assert sort(nums) == sorted(nums)
    assert sort([-5, 0, 5, -1000, 1000]) == [-1000, -5, 0, 5, 1000]


@pytest.mark.parametrize("sort", [merge_sort, quick_sort, heap_sort], ids=lambda f: f.__name__)
def test_large_inputs_including_adversarial_patterns(sort):
    rnd = random.Random(1)
    n = 20_000
    for nums in (
        [rnd.randint(0, 10**9) for _ in range(n)],
        list(range(n)),  # already sorted: kills naive first-element-pivot quicksort
        list(range(n, 0, -1)),
        [5] * n,  # all equal: kills 2-way-partition quicksort
        [rnd.randint(0, 3) for _ in range(n)],
    ):
        assert sort(nums) == sorted(nums)


def test_quick_sort_stack_depth_stays_small():
    # Recursing only into the smaller side keeps depth O(log n) even at the default limit.
    assert sys.getrecursionlimit() <= 10_000
    nums = list(range(100_000))
    assert quick_sort(nums) == nums


def test_radix_sort_other_bases_and_big_numbers():
    rnd = random.Random(2)
    nums = [rnd.randint(-(10**12), 10**12) for _ in range(500)]
    for base in (2, 10, 256):
        assert radix_sort(nums, base=base) == sorted(nums)


def test_counting_sort_handles_offset_range():
    assert counting_sort([1_000_003, 1_000_001, 1_000_002]) == [1_000_001, 1_000_002, 1_000_003]
