"""Interval DP and bitmask DP."""

from __future__ import annotations

from functools import cache

# --- Interval DP -------------------------------------------------------------------
# State = a contiguous range [i, j]. Solve short ranges first; a longer range is
# solved by trying every place to split it (or every "last action" inside it).


def matrix_chain_order(dims: list[int]) -> int:
    """Fewest scalar multiplications to multiply a chain of matrices.

    Matrix k has shape dims[k] x dims[k+1]. cost[i][j] = cheapest way to
    multiply matrices i..j = min over the LAST multiplication's split point k:
        cost[i][k] + cost[k+1][j] + dims[i] * dims[k+1] * dims[j+1]
    """
    n = len(dims) - 1  # number of matrices
    if n <= 1:
        return 0
    cost = [[0] * n for _ in range(n)]
    for length in range(2, n + 1):  # short chains first
        for i in range(n - length + 1):
            j = i + length - 1
            cost[i][j] = min(
                cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                for k in range(i, j)
            )
    return cost[0][n - 1]


def burst_balloons(nums: list[int]) -> int:
    """Max coins from bursting all balloons; bursting i earns left * nums[i] * right.

    Trick: think about which balloon in a range is burst LAST. When k is the
    last balloon burst strictly between i and j, its neighbours at that moment
    are exactly balloons i and j, so the two sides become independent.
    """
    vals = [1, *nums, 1]  # virtual balloons worth 1 at both ends
    n = len(vals)
    best = [[0] * n for _ in range(n)]  # best[i][j]: burst everything strictly between i and j
    for gap in range(2, n):
        for i in range(n - gap):
            j = i + gap
            best[i][j] = max(
                best[i][k] + best[k][j] + vals[i] * vals[k] * vals[j] for k in range(i + 1, j)
            )
    return best[0][n - 1]


def min_palindrome_cuts(s: str) -> int:
    """Fewest cuts so every piece of s is a palindrome.

    Step 1 (interval DP): pal[i][j] says whether s[i..j] is a palindrome:
    the ends match and the inside is a palindrome.
    Step 2 (1D DP): cuts[j] = fewest cuts for the prefix s[..j].
    """
    n = len(s)
    if n == 0:
        return 0
    pal = [[False] * n for _ in range(n)]
    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            pal[i][j] = s[i] == s[j] and (length <= 2 or pal[i + 1][j - 1])
    cuts = [0] * n
    for j in range(n):
        if pal[0][j]:
            cuts[j] = 0
        else:
            cuts[j] = min(cuts[i - 1] + 1 for i in range(1, j + 1) if pal[i][j])
    return cuts[n - 1]


def longest_palindromic_substring(s: str) -> str:
    """Longest contiguous palindrome, by expanding around each of the 2n - 1 centres."""
    best_lo, best_hi = 0, 0  # s[best_lo:best_hi]
    for centre in range(2 * len(s) - 1):
        lo, hi = centre // 2, (centre + 1) // 2  # same index (odd) or neighbours (even)
        while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
            lo, hi = lo - 1, hi + 1
        if hi - lo - 1 > best_hi - best_lo:
            best_lo, best_hi = lo + 1, hi
    return s[best_lo:best_hi]


# --- Bitmask DP ---------------------------------------------------------------------
# State includes a SET of items packed into the bits of an integer: bit i set
# means item i is "used". Works when n is small (about 20 or fewer): 2^n states.


def tsp(dist: list[list[float]]) -> float:
    """Travelling salesman (Held-Karp): shortest tour visiting every city once
    and returning to city 0. O(n^2 * 2^n) time instead of O(n!).

    best[mask][last] = shortest path that starts at city 0, visits exactly the
    cities in `mask`, and currently stands at `last`.
    """
    n = len(dist)
    if n <= 1:
        return 0
    INF = float("inf")
    best = [[INF] * n for _ in range(1 << n)]
    best[1][0] = 0  # only city 0 visited, standing at city 0
    for mask in range(1, 1 << n):
        if not mask & 1:
            continue  # every path starts at city 0
        for last in range(n):
            cur = best[mask][last]
            if cur == INF:
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue  # already visited
                new_mask = mask | (1 << nxt)
                if cur + dist[last][nxt] < best[new_mask][nxt]:
                    best[new_mask][nxt] = cur + dist[last][nxt]
    full = (1 << n) - 1
    return min(best[full][last] + dist[last][0] for last in range(1, n))


def min_cost_assignment(cost: list[list[int]]) -> int:
    """Assign n workers to n jobs (one each) at minimum total cost. O(n * 2^n).

    Workers are assigned in order, so the number of set bits in `mask` (jobs
    already taken) tells us which worker is next; no extra state is needed.
    """
    n = len(cost)
    INF = float("inf")
    best = [INF] * (1 << n)
    best[0] = 0
    for mask in range(1 << n):
        if best[mask] == INF:
            continue
        worker = bin(mask).count("1")
        if worker == n:
            continue
        for job in range(n):
            if not mask & (1 << job):
                new_mask = mask | (1 << job)
                best[new_mask] = min(best[new_mask], best[mask] + cost[worker][job])
    return int(best[(1 << n) - 1]) if n else 0


def can_partition_k_subsets(nums: list[int], k: int) -> bool:
    """Can nums be split into k groups with equal sums? Top-down bitmask DP.

    Fill one group at a time. The state is just the set of used numbers: the
    fill level of the current group is (sum of used numbers) mod target.
    """
    total = sum(nums)
    if k <= 0 or total % k:
        return False
    target = total // k
    if any(x > target for x in nums):
        return False
    n = len(nums)

    @cache
    def solve(mask: int, filled: int) -> bool:
        if mask == (1 << n) - 1:
            return True
        for i in range(n):
            if mask & (1 << i) or filled + nums[i] > target:
                continue
            if solve(mask | (1 << i), (filled + nums[i]) % target):
                return True
        return False

    return solve(0, 0)
