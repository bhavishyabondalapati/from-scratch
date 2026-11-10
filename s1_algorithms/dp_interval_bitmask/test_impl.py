import random
from functools import lru_cache
from itertools import permutations

from .impl import (
    burst_balloons,
    can_partition_k_subsets,
    longest_palindromic_substring,
    matrix_chain_order,
    min_cost_assignment,
    min_palindrome_cuts,
    tsp,
)


def test_matrix_chain_against_recursive_reference():
    rnd = random.Random(0)

    def ref(dims):
        @lru_cache(maxsize=None)
        def go(i, j):
            if i == j:
                return 0
            return min(
                go(i, k) + go(k + 1, j) + dims[i] * dims[k + 1] * dims[j + 1] for k in range(i, j)
            )

        return go(0, len(dims) - 2) if len(dims) >= 2 else 0

    for _ in range(100):
        dims = [rnd.randint(1, 12) for _ in range(rnd.randint(1, 8))]
        assert matrix_chain_order(dims) == ref(tuple(dims))
    assert matrix_chain_order([10, 30, 5, 60]) == 4500  # (AB)C: 1500 + 3000


def test_burst_balloons_against_brute_force():
    rnd = random.Random(1)

    def brute(nums):
        if not nums:
            return 0
        best = 0
        for i in range(len(nums)):
            left = nums[i - 1] if i > 0 else 1
            right = nums[i + 1] if i + 1 < len(nums) else 1
            best = max(best, left * nums[i] * right + brute(nums[:i] + nums[i + 1 :]))
        return best

    for _ in range(100):
        nums = tuple(rnd.randint(0, 9) for _ in range(rnd.randint(0, 6)))
        assert burst_balloons(list(nums)) == brute(nums)
    assert burst_balloons([3, 1, 5, 8]) == 167


def test_min_palindrome_cuts_against_brute_force():
    rnd = random.Random(2)

    @lru_cache(maxsize=None)
    def brute(s):
        if s == s[::-1]:
            return 0
        return min(1 + brute(s[i:]) for i in range(1, len(s)) if s[:i] == s[:i][::-1])

    for _ in range(300):
        s = "".join(rnd.choice("ab") for _ in range(rnd.randint(0, 10)))
        assert min_palindrome_cuts(s) == brute(s)
    assert min_palindrome_cuts("aab") == 1


def test_longest_palindromic_substring_against_brute_force():
    rnd = random.Random(3)
    for _ in range(300):
        s = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 12)))
        best = max(
            (j - i for i in range(len(s)) for j in range(i + 1, len(s) + 1) if s[i:j] == s[i:j][::-1]),
            default=0,
        )
        got = longest_palindromic_substring(s)
        assert got in s and got == got[::-1] and len(got) == best


def test_tsp_against_all_permutations():
    rnd = random.Random(4)
    for _ in range(60):
        n = rnd.randint(1, 7)
        dist = [[0 if i == j else rnd.randint(1, 30) for j in range(n)] for i in range(n)]  # asymmetric
        brute = min(
            (
                sum(dist[a][b] for a, b in zip((0, *order), (*order, 0)))
                for order in permutations(range(1, n))
            ),
        )
        assert tsp(dist) == brute


def test_assignment_against_all_permutations():
    rnd = random.Random(5)
    for _ in range(100):
        n = rnd.randint(0, 6)
        cost = [[rnd.randint(1, 20) for _ in range(n)] for _ in range(n)]
        brute = min(sum(cost[w][j] for w, j in enumerate(p)) for p in permutations(range(n)))
        assert min_cost_assignment(cost) == brute


def test_partition_k_subsets_against_brute_force():
    rnd = random.Random(6)

    def brute(nums, k):
        total = sum(nums)
        if total % k:
            return False
        target = total // k

        def place(i, bins):
            if i == len(nums):
                return all(b == target for b in bins)
            for b in range(k):
                if bins[b] + nums[i] <= target:
                    bins[b] += nums[i]
                    if place(i + 1, bins):
                        return True
                    bins[b] -= nums[i]
            return False

        return place(0, [0] * k)

    seen = set()
    for _ in range(200):
        nums = [rnd.randint(1, 6) for _ in range(rnd.randint(1, 8))]
        k = rnd.randint(1, 4)
        result = can_partition_k_subsets(nums, k)
        assert result == brute(nums, k)
        seen.add(result)
    assert seen == {True, False}  # both outcomes were exercised
