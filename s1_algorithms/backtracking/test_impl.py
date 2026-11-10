import copy
import itertools
import random

from .impl import (
    combination_sum,
    combinations,
    permutations,
    solve_n_queens,
    solve_sudoku,
    subsets,
    subsets_with_duplicates,
)


def test_subsets_match_itertools():
    for n in range(0, 7):
        nums = list(range(n))
        expected = {c for k in range(n + 1) for c in itertools.combinations(nums, k)}
        got = [tuple(s) for s in subsets(nums)]
        assert len(got) == 2**n and set(got) == expected


def test_subsets_with_duplicates():
    rnd = random.Random(0)
    for _ in range(100):
        nums = [rnd.randint(0, 3) for _ in range(rnd.randint(0, 7))]
        expected = {
            tuple(sorted(c)) for k in range(len(nums) + 1) for c in itertools.combinations(nums, k)
        }
        got = [tuple(s) for s in subsets_with_duplicates(nums)]
        assert len(got) == len(set(got))  # no repeats
        assert set(got) == expected


def test_permutations_match_itertools():
    for n in range(0, 6):
        nums = list(range(n))
        assert [tuple(p) for p in permutations(nums)] == list(itertools.permutations(nums))


def test_combinations_match_itertools():
    for n in range(0, 8):
        for k in range(0, n + 2):
            expected = [list(c) for c in itertools.combinations(range(1, n + 1), k)]
            assert combinations(n, k) == expected


def test_combination_sum_against_brute_force():
    rnd = random.Random(1)
    for _ in range(100):
        candidates = rnd.sample(range(2, 10), rnd.randint(1, 4))
        target = rnd.randint(1, 18)
        expected = set()
        for size in range(1, target // min(candidates) + 1):
            for combo in itertools.combinations_with_replacement(sorted(candidates), size):
                if sum(combo) == target:
                    expected.add(combo)
        got = [tuple(c) for c in combination_sum(candidates, target)]
        assert len(got) == len(set(got)) and set(got) == expected


def test_n_queens_counts_and_validity():
    # Known number of solutions for n = 1..8 (OEIS A000170).
    for n, count in zip(range(1, 9), [1, 0, 0, 2, 10, 4, 40, 92]):
        solutions = solve_n_queens(n)
        assert len(solutions) == count
        for cols in solutions:
            assert sorted(cols) == list(range(n))  # one queen per column
            assert len({r - c for r, c in enumerate(cols)}) == n  # no shared "\" diagonal
            assert len({r + c for r, c in enumerate(cols)}) == n  # no shared "/" diagonal


PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]


def assert_valid_solution(board, puzzle):
    digits = set(range(1, 10))
    for i in range(9):
        assert set(board[i]) == digits
        assert {board[r][i] for r in range(9)} == digits
        br, bc = 3 * (i // 3), 3 * (i % 3)
        assert {board[br + r][bc + c] for r in range(3) for c in range(3)} == digits
    for r in range(9):
        for c in range(9):
            assert puzzle[r][c] in (0, board[r][c])  # clues untouched


def test_sudoku_solves_puzzle_and_empty_board():
    board = copy.deepcopy(PUZZLE)
    assert solve_sudoku(board)
    assert_valid_solution(board, PUZZLE)
    assert board[0] == [5, 3, 4, 6, 7, 8, 9, 1, 2]  # this puzzle has a unique solution

    empty = [[0] * 9 for _ in range(9)]
    assert solve_sudoku(empty)
    assert_valid_solution(empty, [[0] * 9 for _ in range(9)])


def test_sudoku_detects_unsolvable_boards():
    conflict = copy.deepcopy(PUZZLE)
    conflict[0][2] = 5  # two 5s in the first row
    assert not solve_sudoku(conflict)

    # No direct conflict, but cell (0, 0) has no legal digit: row has 1-6, column has 7-9.
    dead_end = [[0] * 9 for _ in range(9)]
    dead_end[0][3:9] = [1, 2, 3, 4, 5, 6]
    dead_end[3][0], dead_end[4][0], dead_end[5][0] = 7, 8, 9
    before = copy.deepcopy(dead_end)
    assert not solve_sudoku(dead_end)
    assert dead_end == before  # board restored after failing
