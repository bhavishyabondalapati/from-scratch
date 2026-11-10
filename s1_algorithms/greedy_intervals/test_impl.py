import random
from itertools import combinations

import numpy as np
import pytest
from scipy.optimize import linprog

from .impl import (
    can_jump,
    fractional_knapsack,
    gas_station,
    insert_interval,
    max_non_overlapping,
    merge_intervals,
    min_arrows,
    min_jumps,
    min_meeting_rooms,
    min_removals_for_no_overlap,
)
from .impl import (
    merge_intervals as merge,
)


def rand_intervals(rnd, max_n=8, span=15):
    out = []
    for _ in range(rnd.randint(0, max_n)):
        a = rnd.randint(0, span)
        out.append((a, a + rnd.randint(1, 5)))
    return out


def covered_points(intervals):
    """Half-integer grid points covered by closed intervals (detects touching vs gaps)."""
    return {x / 2 for a, b in intervals for x in range(2 * a, 2 * b + 1)}


def test_merge_intervals_covers_same_points_and_is_disjoint():
    rnd = random.Random(0)
    for _ in range(300):
        ivs = rand_intervals(rnd)
        merged = merge_intervals(ivs)
        assert covered_points(merged) == covered_points(ivs)
        assert all(a[1] < b[0] for a, b in zip(merged, merged[1:]))  # sorted, strictly apart
    assert merge_intervals([(1, 3), (2, 6), (8, 10), (10, 12)]) == [(1, 6), (8, 12)]


def test_insert_interval_matches_merge_of_everything():
    rnd = random.Random(1)
    for _ in range(300):
        base = merge(rand_intervals(rnd))
        a = rnd.randint(0, 20)
        new = (a, a + rnd.randint(0, 6))
        assert insert_interval(base, new) == merge([*base, new])


def conflict(a, b):
    return a[0] < b[1] and b[0] < a[1]


def test_max_non_overlapping_against_brute_force():
    rnd = random.Random(2)
    for _ in range(200):
        ivs = rand_intervals(rnd)
        best = max(
            k
            for k in range(len(ivs) + 1)
            for subset in combinations(ivs, k)
            if not any(conflict(a, b) for a, b in combinations(subset, 2))
        )
        assert max_non_overlapping(ivs) == best
        assert min_removals_for_no_overlap(ivs) == len(ivs) - best


def test_min_meeting_rooms_equals_peak_overlap():
    rnd = random.Random(3)
    for _ in range(300):
        ivs = rand_intervals(rnd)
        # Meetings are half-open [start, end): count how many are running at each instant.
        peak = max((sum(a <= t < b for a, b in ivs) for t in range(0, 25)), default=0)
        assert min_meeting_rooms(ivs) == peak
    assert min_meeting_rooms([(0, 30), (5, 10), (15, 20)]) == 2


def test_min_arrows_against_brute_force():
    rnd = random.Random(4)
    for _ in range(150):
        ivs = rand_intervals(rnd, max_n=6, span=10)
        points = sorted({b for _, b in ivs})  # an optimal solution only needs end points
        best = next(
            k
            for k in range(len(ivs) + 1)
            for shots in combinations(points, k)
            if all(any(a <= p <= b for p in shots) for a, b in ivs)
        )
        assert min_arrows(ivs) == best


def test_jump_game_against_bfs():
    rnd = random.Random(5)
    for _ in range(400):
        nums = [rnd.randint(0, 3) for _ in range(rnd.randint(1, 12))]
        n = len(nums)
        dist = [None] * n
        dist[0] = 0
        for i in range(n):  # simple forward DP as the reference
            if dist[i] is None:
                continue
            for j in range(i + 1, min(n, i + nums[i] + 1)):
                if dist[j] is None or dist[i] + 1 < dist[j]:
                    dist[j] = dist[i] + 1
        assert can_jump(nums) == (dist[-1] is not None)
        assert min_jumps(nums) == (-1 if dist[-1] is None else dist[-1])


def test_gas_station_against_simulation():
    rnd = random.Random(6)

    def works(gas, cost, s):
        tank = 0
        for k in range(len(gas)):
            i = (s + k) % len(gas)
            tank += gas[i] - cost[i]
            if tank < 0:
                return False
        return True

    for _ in range(400):
        n = rnd.randint(1, 8)
        gas = [rnd.randint(0, 6) for _ in range(n)]
        cost = [rnd.randint(0, 6) for _ in range(n)]
        got = gas_station(gas, cost)
        valid = [s for s in range(n) if works(gas, cost, s)]
        if not valid:
            assert got == -1
        else:
            assert got in valid


def test_fractional_knapsack_matches_linear_program():
    rnd = random.Random(7)
    for _ in range(100):
        n = rnd.randint(1, 8)
        weights = [rnd.randint(1, 10) for _ in range(n)]
        values = [rnd.randint(1, 30) for _ in range(n)]
        capacity = rnd.randint(0, 30)
        # maximise v.x  subject to  w.x <= capacity,  0 <= x_i <= 1
        lp = linprog(-np.array(values), A_ub=[weights], b_ub=[capacity], bounds=[(0, 1)] * n)
        assert fractional_knapsack(weights, values, capacity) == pytest.approx(-lp.fun, abs=1e-6)
