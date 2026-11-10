import random
from functools import lru_cache
from itertools import combinations, product

from .impl import (
    can_partition,
    coin_change_permutations,
    coin_change_ways,
    knapsack_01,
    knapsack_01_items,
    knapsack_01_table,
    knapsack_unbounded,
    subset_sum,
    target_sum_ways,
)


def random_items(rnd, max_n=8):
    n = rnd.randint(0, max_n)
    return [rnd.randint(1, 10) for _ in range(n)], [rnd.randint(1, 30) for _ in range(n)]


def test_knapsack_01_against_brute_force():
    rnd = random.Random(0)
    for _ in range(200):
        weights, values = random_items(rnd)
        capacity = rnd.randint(0, 30)
        n = len(weights)
        best = 0
        for mask in product([0, 1], repeat=n):  # every subset
            if sum(w for w, m in zip(weights, mask) if m) <= capacity:
                best = max(best, sum(v for v, m in zip(values, mask) if m))
        assert knapsack_01(weights, values, capacity) == best
        assert knapsack_01_table(weights, values, capacity) == best
        chosen = knapsack_01_items(weights, values, capacity)
        assert len(set(chosen)) == len(chosen)
        assert sum(weights[i] for i in chosen) <= capacity
        assert sum(values[i] for i in chosen) == best


def test_knapsack_unbounded_against_recursive_reference():
    rnd = random.Random(1)
    for _ in range(200):
        weights, values = random_items(rnd, max_n=5)
        capacity = rnd.randint(0, 30)

        @lru_cache(maxsize=None)
        def best(c, items=tuple(zip(weights, values))):
            return max((v + best(c - w) for w, v in items if w <= c), default=0)

        assert knapsack_unbounded(weights, values, capacity) == best(capacity)
        # Being allowed to reuse items can never be worse than using each once.
        assert knapsack_unbounded(weights, values, capacity) >= knapsack_01(
            weights, values, capacity
        )


def test_subset_sum_and_partition_against_brute_force():
    rnd = random.Random(2)
    for _ in range(300):
        nums = [rnd.randint(0, 12) for _ in range(rnd.randint(0, 8))]
        sums = {sum(c) for k in range(len(nums) + 1) for c in combinations(nums, k)}
        target = rnd.randint(0, 40)
        assert subset_sum(nums, target) == (target in sums)
        assert can_partition(nums) == (sum(nums) % 2 == 0 and sum(nums) // 2 in sums)


def test_coin_change_counts_against_brute_force():
    rnd = random.Random(3)
    for _ in range(100):
        coins = rnd.sample(range(1, 8), rnd.randint(1, 3))
        amount = rnd.randint(0, 14)

        @lru_cache(maxsize=None)
        def ordered(a, coins=tuple(coins)):
            return 1 if a == 0 else sum(ordered(a - c) for c in coins if c <= a)

        @lru_cache(maxsize=None)
        def unordered(a, i, coins=tuple(coins)):
            # Combinations using only coins[i:], so each multiset is counted once.
            if a == 0:
                return 1
            if i == len(coins):
                return 0
            return sum(unordered(a - k * coins[i], i + 1) for k in range(a // coins[i] + 1))

        assert coin_change_permutations(coins, amount) == ordered(amount)
        assert coin_change_ways(coins, amount) == unordered(amount, 0)
    assert coin_change_ways([1, 2, 5], 5) == 4
    assert coin_change_permutations([1, 2, 5], 5) == 9


def test_target_sum_ways_against_brute_force():
    rnd = random.Random(4)
    for _ in range(200):
        nums = [rnd.randint(0, 6) for _ in range(rnd.randint(0, 8))]
        target = rnd.randint(-15, 15)
        expected = sum(
            sum(s * x for s, x in zip(signs, nums)) == target
            for signs in product([1, -1], repeat=len(nums))
        )
        assert target_sum_ways(nums, target) == expected
