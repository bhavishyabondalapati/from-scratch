import collections
import random

import pytest

from .impl import Deque, QueueWithTwoStacks, shortest_subarray_sum_at_least, sliding_window_max


def test_deque_matches_collections_deque():
    rnd = random.Random(0)
    ours, ref = Deque(capacity=2), collections.deque()
    for step in range(6000):
        op = rnd.choice(["append", "appendleft", "pop", "popleft"])
        if op.startswith("append"):
            getattr(ours, op)(step)
            getattr(ref, op)(step)
        elif ref:
            assert getattr(ours, op)() == getattr(ref, op)()
        else:
            with pytest.raises(IndexError):
                getattr(ours, op)()
        assert len(ours) == len(ref)
        if ref:
            i = rnd.randrange(len(ref))
            assert ours[i] == ref[i] and ours[-1] == ref[-1] and ours[0] == ref[0]


def test_queue_with_two_stacks_is_fifo():
    rnd = random.Random(1)
    ours, ref = QueueWithTwoStacks(), collections.deque()
    for step in range(3000):
        if ref and rnd.random() < 0.5:
            assert ours.peek() == ref[0]
            assert ours.pop() == ref.popleft()
        else:
            ours.push(step)
            ref.append(step)
        assert len(ours) == len(ref)


def test_sliding_window_max_against_brute_force():
    rnd = random.Random(2)
    for _ in range(300):
        nums = [rnd.randint(-9, 9) for _ in range(rnd.randint(1, 20))]
        k = rnd.randint(1, len(nums))
        assert sliding_window_max(nums, k) == [
            max(nums[i : i + k]) for i in range(len(nums) - k + 1)
        ]


def test_shortest_subarray_with_negatives_against_brute_force():
    rnd = random.Random(3)
    for _ in range(400):
        nums = [rnd.randint(-5, 8) for _ in range(rnd.randint(1, 12))]
        k = rnd.randint(1, 20)
        n = len(nums)
        expected = min(
            (j - i for i in range(n) for j in range(i + 1, n + 1) if sum(nums[i:j]) >= k),
            default=-1,
        )
        assert shortest_subarray_sum_at_least(nums, k) == expected
