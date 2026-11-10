import heapq
import random
import statistics

import pytest

from .impl import Heap, MedianFinder, merge_k_sorted, top_k_largest


def is_heap(a, key=lambda x: x):
    return all(key(a[(i - 1) // 2]) <= key(a[i]) for i in range(1, len(a)))


def test_heap_matches_heapq_under_random_operations():
    rnd = random.Random(0)
    ours, ref = Heap(), []
    for _ in range(5000):
        if ref and rnd.random() < 0.45:
            assert ours.pop() == heapq.heappop(ref)
        else:
            x = rnd.randint(-100, 100)
            ours.push(x)
            heapq.heappush(ref, x)
        assert len(ours) == len(ref)
        assert is_heap(ours._a)
        if ref:
            assert ours.peek() == ref[0]


def test_heapify_and_heap_sort():
    rnd = random.Random(1)
    for _ in range(200):
        xs = [rnd.randint(-50, 50) for _ in range(rnd.randint(0, 40))]
        h = Heap(xs)
        assert is_heap(h._a)
        assert [h.pop() for _ in range(len(xs))] == sorted(xs)
        mx = Heap(xs, key=lambda x: -x)  # max-heap via key
        assert [mx.pop() for _ in range(len(xs))] == sorted(xs, reverse=True)


def test_empty_heap_raises():
    with pytest.raises(IndexError):
        Heap().pop()
    with pytest.raises(IndexError):
        Heap().peek()


def test_top_k_largest_matches_heapq():
    rnd = random.Random(2)
    for _ in range(300):
        xs = [rnd.randint(-50, 50) for _ in range(rnd.randint(0, 30))]
        k = rnd.randint(0, 35)
        assert top_k_largest(xs, k) == heapq.nlargest(k, xs)


def test_merge_k_sorted_matches_heapq_merge():
    rnd = random.Random(3)
    for _ in range(300):
        lists = [
            sorted(rnd.randint(-50, 50) for _ in range(rnd.randint(0, 8)))
            for _ in range(rnd.randint(0, 6))
        ]
        assert merge_k_sorted(lists) == list(heapq.merge(*lists))


def test_median_finder_matches_statistics_median():
    rnd = random.Random(4)
    mf, seen = MedianFinder(), []
    with pytest.raises(ValueError):
        mf.median()
    for _ in range(1500):
        x = rnd.randint(-1000, 1000)
        mf.add(x)
        seen.append(x)
        assert mf.median() == statistics.median(seen)
