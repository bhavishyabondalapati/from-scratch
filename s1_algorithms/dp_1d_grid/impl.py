"""Dynamic programming: 1D problems and 2D grid problems.

Recipe: (1) define the state in words, (2) write the recurrence, (3) set base
cases, (4) pick an evaluation order so every dependency is ready, (5) shrink
memory if only the last row / last few values are needed.
"""

from __future__ import annotations

from functools import cache

# --- 1D ------------------------------------------------------------------------


def fib_memo(n: int) -> int:
    """Top-down: plain recursion plus a cache (memoization)."""

    @cache
    def f(k: int) -> int:
        return k if k < 2 else f(k - 1) + f(k - 2)

    return f(n)


def fib_tab(n: int) -> int:
    """Bottom-up with O(1) memory: only the last two values are ever needed."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def climb_stairs(n: int) -> int:
    """Ways to climb n steps taking 1 or 2 at a time. ways[i] = ways[i-1] + ways[i-2]."""
    prev, cur = 1, 1  # ways[0], ways[1]
    for _ in range(n - 1):
        prev, cur = cur, prev + cur
    return cur if n >= 1 else 1


def house_robber(values: list[int]) -> int:
    """Max sum of non-adjacent, non-negative values.

    best[i] = max(best[i-1],            # skip house i
                  best[i-2] + values[i])  # rob house i
    """
    skip, take = 0, 0  # best[i-2], best[i-1]
    for v in values:
        skip, take = take, max(take, skip + v)
    return take


def max_subarray(nums: list[int]) -> int:
    """Kadane's algorithm: largest sum of a non-empty contiguous subarray.

    State: best sum of a subarray ENDING at i. Either extend the previous
    subarray or start fresh at i.
    """
    best = ending_here = nums[0]
    for x in nums[1:]:
        ending_here = max(x, ending_here + x)
        best = max(best, ending_here)
    return best


def coin_change(coins: list[int], amount: int) -> int:
    """Fewest coins to make `amount` (unlimited supply), or -1.

    dp[a] = 1 + min(dp[a - c] for each coin c <= a)
    """
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return dp[amount] if dp[amount] < INF else -1


def word_break(s: str, words: list[str]) -> bool:
    """Can s be cut into a sequence of dictionary words?

    ok[i] = True if s[:i] can be segmented = any j < i with ok[j] and s[j:i] a word.
    """
    vocab = set(words)
    longest = max(map(len, vocab), default=0)
    ok = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(max(0, i - longest), i):
            if ok[j] and s[j:i] in vocab:
                ok[i] = True
                break
    return ok[len(s)]


def decode_ways(s: str) -> int:
    """Number of ways to decode digits where 'A'=1 ... 'Z'=26.

    ways[i] = (ways[i-1] if s[i-1] is 1-9) + (ways[i-2] if s[i-2:i] is 10-26)
    """
    prev, cur = 0, 1  # ways[i-2], ways[i-1]; the empty string has one decoding
    for i in range(len(s)):
        nxt = 0
        if s[i] != "0":
            nxt += cur
        if i >= 1 and "10" <= s[i - 1 : i + 1] <= "26":
            nxt += prev
        prev, cur = cur, nxt
    return cur


# --- 2D grid ----------------------------------------------------------------------


def unique_paths(rows: int, cols: int, blocked: set[tuple[int, int]] | None = None) -> int:
    """Paths from top-left to bottom-right moving only right or down.

    paths[r][c] = paths[r-1][c] + paths[r][c-1], and 0 on blocked cells.
    One row of memory is enough: `row[c]` still holds the value from the row
    above when we read it, and `row[c-1]` is already the current row.
    """
    blocked = blocked or set()
    row = [0] * cols
    row[0] = 1
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked:
                row[c] = 0
            elif c > 0:
                row[c] += row[c - 1]
    return row[-1]


def min_path_sum(grid: list[list[int]]) -> int:
    """Cheapest top-left -> bottom-right path moving right or down."""
    rows, cols = len(grid), len(grid[0])
    cost = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                best_prev = 0
            elif r == 0:
                best_prev = cost[r][c - 1]
            elif c == 0:
                best_prev = cost[r - 1][c]
            else:
                best_prev = min(cost[r - 1][c], cost[r][c - 1])
            cost[r][c] = grid[r][c] + best_prev
    return cost[-1][-1]


def maximal_square(matrix: list[list[int]]) -> int:
    """Area of the largest all-ones square in a 0/1 matrix.

    side[r][c] = side of the largest square whose BOTTOM-RIGHT corner is (r, c)
               = 1 + min(up, left, up-left) when matrix[r][c] == 1.
    The square can only be as big as its three neighbours allow.
    """
    if not matrix:
        return 0
    rows, cols = len(matrix), len(matrix[0])
    side = [[0] * (cols + 1) for _ in range(rows + 1)]  # padded with a zero border
    best = 0
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if matrix[r - 1][c - 1] == 1:
                side[r][c] = 1 + min(side[r - 1][c], side[r][c - 1], side[r - 1][c - 1])
                best = max(best, side[r][c])
    return best * best


def triangle_min_path(triangle: list[list[int]]) -> int:
    """Min top-to-bottom path sum in a triangle, moving to an adjacent number below.

    Work bottom-up so each cell just picks the better of its two children.
    """
    best = list(triangle[-1])
    for row in reversed(triangle[:-1]):
        best = [x + min(best[i], best[i + 1]) for i, x in enumerate(row)]
    return best[0]
