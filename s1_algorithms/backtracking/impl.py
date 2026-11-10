"""Backtracking: subsets, permutations, combinations, N-Queens, sudoku.

Template: choose -> explore -> un-choose. Build a partial answer one decision at a
time and abandon ("prune") it as soon as it cannot lead to a valid full answer.
"""

from __future__ import annotations


def subsets(nums: list[int]) -> list[list[int]]:
    """All 2^n subsets. At each index decide: take it or skip it."""
    out: list[list[int]] = []
    current: list[int] = []

    def explore(i: int) -> None:
        if i == len(nums):
            out.append(current.copy())  # copy! `current` keeps changing
            return
        explore(i + 1)  # skip nums[i]
        current.append(nums[i])  # choose
        explore(i + 1)  # explore
        current.pop()  # un-choose

    explore(0)
    return out


def subsets_with_duplicates(nums: list[int]) -> list[list[int]]:
    """All distinct subsets when nums may contain repeated values."""
    nums = sorted(nums)
    out: list[list[int]] = []
    current: list[int] = []

    def explore(start: int) -> None:
        out.append(current.copy())
        for i in range(start, len(nums)):
            # Same value as the previous candidate AT THIS DEPTH -> same subtree; skip.
            if i > start and nums[i] == nums[i - 1]:
                continue
            current.append(nums[i])
            explore(i + 1)
            current.pop()

    explore(0)
    return out


def permutations(nums: list[int]) -> list[list[int]]:
    """All n! orderings of distinct values."""
    out: list[list[int]] = []
    current: list[int] = []
    used = [False] * len(nums)

    def explore() -> None:
        if len(current) == len(nums):
            out.append(current.copy())
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            current.append(x)
            explore()
            current.pop()
            used[i] = False

    explore()
    return out


def combinations(n: int, k: int) -> list[list[int]]:
    """All k-element subsets of 1..n, in lexicographic order."""
    out: list[list[int]] = []
    current: list[int] = []

    def explore(start: int) -> None:
        if len(current) == k:
            out.append(current.copy())
            return
        need = k - len(current)
        # Prune: stop when there are not enough numbers left to fill the combination.
        for x in range(start, n - need + 2):
            current.append(x)
            explore(x + 1)
            current.pop()

    explore(1)
    return out


def combination_sum(candidates: list[int], target: int) -> list[list[int]]:
    """Combinations of positive candidates (reuse allowed) that sum to target."""
    candidates = sorted(set(candidates))
    out: list[list[int]] = []
    current: list[int] = []

    def explore(start: int, remaining: int) -> None:
        if remaining == 0:
            out.append(current.copy())
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break  # sorted, so everything after is too big as well
            current.append(candidates[i])
            explore(i, remaining - candidates[i])  # i, not i + 1: reuse is allowed
            current.pop()

    explore(0, target)
    return out


def solve_n_queens(n: int) -> list[list[int]]:
    """All ways to place n non-attacking queens. Each solution lists the queen's
    column for every row.

    Place one queen per row. A square (r, c) is attacked if its column, its
    "\\" diagonal (r - c is constant) or its "/" diagonal (r + c is constant)
    is already taken; three sets make that check O(1).
    """
    out: list[list[int]] = []
    cols: list[int] = []
    used_cols: set[int] = set()
    used_diag: set[int] = set()
    used_anti: set[int] = set()

    def place(r: int) -> None:
        if r == n:
            out.append(cols.copy())
            return
        for c in range(n):
            if c in used_cols or (r - c) in used_diag or (r + c) in used_anti:
                continue
            cols.append(c)
            used_cols.add(c)
            used_diag.add(r - c)
            used_anti.add(r + c)
            place(r + 1)
            cols.pop()
            used_cols.remove(c)
            used_diag.remove(r - c)
            used_anti.remove(r + c)

    place(0)
    return out


def solve_sudoku(board: list[list[int]]) -> bool:
    """Fill a 9x9 board in place (0 = empty). Returns False if unsolvable.

    Heuristic: always fill the empty cell with the FEWEST legal digits next
    ("most constrained first"). It fails fast and shrinks the search hugely.
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty: list[tuple[int, int]] = []
    for r in range(9):
        for c in range(9):
            d = board[r][c]
            if d == 0:
                empty.append((r, c))
                continue
            b = (r // 3) * 3 + c // 3
            if d in rows[r] or d in cols[c] or d in boxes[b]:
                return False  # the given clues already conflict
            rows[r].add(d)
            cols[c].add(d)
            boxes[b].add(d)

    def options(r: int, c: int) -> list[int]:
        taken = rows[r] | cols[c] | boxes[(r // 3) * 3 + c // 3]
        return [d for d in range(1, 10) if d not in taken]

    def fill() -> bool:
        if not empty:
            return True
        # Pick the most constrained empty cell and move it to the end of the list.
        best = min(range(len(empty)), key=lambda i: len(options(*empty[i])))
        empty[best], empty[-1] = empty[-1], empty[best]
        r, c = empty.pop()
        b = (r // 3) * 3 + c // 3
        for d in options(r, c):
            board[r][c] = d
            rows[r].add(d)
            cols[c].add(d)
            boxes[b].add(d)
            if fill():
                return True
            rows[r].remove(d)
            cols[c].remove(d)
            boxes[b].remove(d)
        board[r][c] = 0
        empty.append((r, c))  # un-choose the cell itself
        return False

    return fill()
