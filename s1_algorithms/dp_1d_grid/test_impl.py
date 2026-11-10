import math
import random
from functools import cache
from itertools import combinations

from .impl import (
    climb_stairs,
    coin_change,
    decode_ways,
    fib_memo,
    fib_tab,
    house_robber,
    max_subarray,
    maximal_square,
    min_path_sum,
    triangle_min_path,
    unique_paths,
    word_break,
)


def test_fibonacci_and_stairs():
    known = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    for n, f in enumerate(known):
        assert fib_memo(n) == fib_tab(n) == f
    assert fib_tab(90) == 2880067194370816120
    # Ways to climb n stairs is Fibonacci shifted by one.
    assert [climb_stairs(n) for n in range(0, 8)] == [1, 1, 2, 3, 5, 8, 13, 21]


def test_house_robber_against_brute_force():
    rnd = random.Random(0)
    for _ in range(200):
        values = [rnd.randint(0, 20) for _ in range(rnd.randint(0, 10))]
        n = len(values)
        best = 0
        for k in range(n + 1):
            for idx in combinations(range(n), k):
                if all(b - a > 1 for a, b in zip(idx, idx[1:])):
                    best = max(best, sum(values[i] for i in idx))
        assert house_robber(values) == best


def test_max_subarray_against_brute_force():
    rnd = random.Random(1)
    for _ in range(300):
        nums = [rnd.randint(-9, 9) for _ in range(rnd.randint(1, 12))]
        n = len(nums)
        expected = max(sum(nums[i:j]) for i in range(n) for j in range(i + 1, n + 1))
        assert max_subarray(nums) == expected


def test_coin_change_against_recursive_reference():
    rnd = random.Random(2)
    for _ in range(100):
        coins = rnd.sample(range(1, 12), rnd.randint(1, 4))
        amount = rnd.randint(0, 40)

        @cache
        def best(a, coins=tuple(coins)):
            if a == 0:
                return 0
            return min((1 + best(a - c) for c in coins if c <= a), default=math.inf)

        expected = best(amount)
        assert coin_change(coins, amount) == (-1 if expected == math.inf else expected)
    assert coin_change([2], 3) == -1 and coin_change([1, 5, 10, 25], 63) == 6


def test_word_break_against_recursive_reference():
    rnd = random.Random(3)
    for _ in range(300):
        words = ["".join(rnd.choice("ab") for _ in range(rnd.randint(1, 3))) for _ in range(3)]
        s = "".join(rnd.choice("ab") for _ in range(rnd.randint(0, 8)))

        def can(rest, words=words):
            return rest == "" or any(rest.startswith(w) and can(rest[len(w) :]) for w in words)

        assert word_break(s, words) == can(s)
    assert word_break("leetcode", ["leet", "code"]) and not word_break("abc", [])


def test_decode_ways_against_recursive_reference():
    rnd = random.Random(4)

    def count(s):
        if s == "":
            return 1
        total = 0
        if s[0] != "0":
            total += count(s[1:])
            if len(s) >= 2 and int(s[:2]) <= 26:
                total += count(s[2:])
        return total

    for _ in range(300):
        s = "".join(rnd.choice("0123") for _ in range(rnd.randint(1, 8)))
        assert decode_ways(s) == count(s)
    assert decode_ways("226") == 3 and decode_ways("06") == 0


def test_unique_paths():
    for rows in range(1, 7):
        for cols in range(1, 7):
            # Without obstacles the answer is "choose which of the moves are downs".
            assert unique_paths(rows, cols) == math.comb(rows + cols - 2, rows - 1)
    assert unique_paths(3, 3, blocked={(1, 1)}) == 2
    assert unique_paths(2, 2, blocked={(0, 0)}) == 0


def grid_paths(rows, cols):
    """Every right/down path through a grid, as lists of cells (brute force)."""
    if rows == 1 and cols == 1:
        return [[(0, 0)]]
    paths = []
    if rows > 1:
        paths += [p + [(rows - 1, cols - 1)] for p in grid_paths(rows - 1, cols)]
    if cols > 1:
        paths += [p + [(rows - 1, cols - 1)] for p in grid_paths(rows, cols - 1)]
    return paths


def test_min_path_sum_against_brute_force():
    rnd = random.Random(5)
    for _ in range(100):
        rows, cols = rnd.randint(1, 5), rnd.randint(1, 5)
        grid = [[rnd.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
        expected = min(sum(grid[r][c] for r, c in p) for p in grid_paths(rows, cols))
        assert min_path_sum(grid) == expected


def test_maximal_square_against_brute_force():
    rnd = random.Random(6)
    for _ in range(200):
        rows, cols = rnd.randint(1, 6), rnd.randint(1, 6)
        m = [[int(rnd.random() < 0.7) for _ in range(cols)] for _ in range(rows)]
        best = 0
        for r in range(rows):
            for c in range(cols):
                for k in range(1, min(rows - r, cols - c) + 1):
                    if all(m[r + i][c + j] for i in range(k) for j in range(k)):
                        best = max(best, k * k)
        assert maximal_square(m) == best
    assert maximal_square([]) == 0


def test_triangle_min_path():
    assert triangle_min_path([[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]) == 11
    rnd = random.Random(7)
    for _ in range(100):
        tri = [[rnd.randint(-5, 9) for _ in range(i + 1)] for i in range(rnd.randint(1, 7))]

        def best(r, c, tri=tri):
            if r == len(tri) - 1:
                return tri[r][c]
            return tri[r][c] + min(best(r + 1, c), best(r + 1, c + 1))

        assert triangle_min_path(tri) == best(0, 0)
