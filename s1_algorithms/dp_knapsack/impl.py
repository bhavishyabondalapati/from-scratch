"""Knapsack dynamic programming: 0/1 and unbounded, plus classic variants."""

from __future__ import annotations


def knapsack_01_table(weights: list[int], values: list[int], capacity: int) -> int:
    """0/1 knapsack with the full 2D table (easiest to understand).

    best[i][c] = max value using only the first i items with capacity c
               = max(best[i-1][c],                      # leave item i
                     best[i-1][c - w_i] + v_i)          # take item i (if it fits)
    """
    n = len(weights)
    best = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            best[i][c] = best[i - 1][c]
            if w <= c:
                best[i][c] = max(best[i][c], best[i - 1][c - w] + v)
    return best[n][capacity]


def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """0/1 knapsack in O(capacity) memory.

    Iterating capacity from HIGH to LOW means best[c - w] is still the value
    from the previous item, so each item is used at most once.
    """
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(capacity, w - 1, -1):
            best[c] = max(best[c], best[c - w] + v)
    return best[capacity]


def knapsack_01_items(weights: list[int], values: list[int], capacity: int) -> list[int]:
    """Indices of one optimal set of items, recovered by walking the table backwards."""
    n = len(weights)
    best = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            best[i][c] = best[i - 1][c]
            if w <= c:
                best[i][c] = max(best[i][c], best[i - 1][c - w] + v)
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if best[i][c] != best[i - 1][c]:  # value changed => item i-1 was taken
            chosen.append(i - 1)
            c -= weights[i - 1]
    return chosen[::-1]


def knapsack_unbounded(weights: list[int], values: list[int], capacity: int) -> int:
    """Unbounded knapsack: each item may be used any number of times.

    Iterating capacity from LOW to HIGH lets best[c - w] already include the
    current item, which is exactly "use it again".
    """
    best = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for c in range(w, capacity + 1):
            best[c] = max(best[c], best[c - w] + v)
    return best[capacity]


def subset_sum(nums: list[int], target: int) -> bool:
    """Is there a subset of non-negative nums that sums to target? (0/1 feasibility)."""
    reachable = [True] + [False] * target
    for x in nums:
        for s in range(target, x - 1, -1):
            reachable[s] = reachable[s] or reachable[s - x]
    return reachable[target]


def can_partition(nums: list[int]) -> bool:
    """Can nums be split into two groups with equal sums?"""
    total = sum(nums)
    return total % 2 == 0 and subset_sum(nums, total // 2)


def coin_change_ways(coins: list[int], amount: int) -> int:
    """Number of COMBINATIONS of coins (order ignored) that make amount.

    Coins in the outer loop: each combination is built in one fixed coin order,
    so [1, 2] and [2, 1] are counted once.
    """
    ways = [1] + [0] * amount
    for coin in coins:
        for a in range(coin, amount + 1):
            ways[a] += ways[a - coin]
    return ways[amount]


def coin_change_permutations(coins: list[int], amount: int) -> int:
    """Number of ORDERED sequences of coins that make amount.

    Amount in the outer loop: for each amount, any coin can be the last one.
    """
    ways = [1] + [0] * amount
    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a:
                ways[a] += ways[a - coin]
    return ways[amount]


def target_sum_ways(nums: list[int], target: int) -> int:
    """Ways to put + or - before each non-negative number so the total is target.

    If P is the sum of the '+' numbers and N the sum of the '-' numbers, then
    P - N = target and P + N = total, so P = (total + target) / 2: count the
    subsets with that sum (a 0/1 counting knapsack).
    """
    total = sum(nums)
    if abs(target) > total or (total + target) % 2:
        return 0
    goal = (total + target) // 2
    ways = [1] + [0] * goal
    for x in nums:
        for s in range(goal, x - 1, -1):
            ways[s] += ways[s - x]
    return ways[goal]
