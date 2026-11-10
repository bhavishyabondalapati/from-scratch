"""Greedy algorithms and interval problems.

Intervals are (start, end) tuples. Unless noted, touching intervals such as
(1, 3) and (3, 5) count as overlapping for merging but NOT as conflicting
meetings (one ends exactly when the next starts).
"""

from __future__ import annotations

import heapq

Interval = tuple[int, int]


def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    """Merge all overlapping intervals. Sort by start, then sweep. O(n log n)."""
    merged: list[Interval] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:  # overlaps the last merged interval
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def insert_interval(intervals: list[Interval], new: Interval) -> list[Interval]:
    """Insert into a sorted, non-overlapping list, merging as needed. O(n)."""
    out: list[Interval] = []
    start, end = new
    i, n = 0, len(intervals)
    while i < n and intervals[i][1] < start:  # entirely before the new interval
        out.append(intervals[i])
        i += 1
    while i < n and intervals[i][0] <= end:  # overlaps: absorb into the new interval
        start, end = min(start, intervals[i][0]), max(end, intervals[i][1])
        i += 1
    out.append((start, end))
    out.extend(intervals[i:])  # entirely after
    return out


def max_non_overlapping(intervals: list[Interval]) -> int:
    """Largest number of mutually non-conflicting intervals (activity selection).

    Greedy: always keep the interval that ENDS first. It leaves the most room
    for everything after it, so it is never a worse choice (exchange argument).
    """
    count, last_end = 0, float("-inf")
    for start, end in sorted(intervals, key=lambda iv: iv[1]):
        if start >= last_end:
            count += 1
            last_end = end
    return count


def min_removals_for_no_overlap(intervals: list[Interval]) -> int:
    """Fewest intervals to delete so the rest do not conflict."""
    return len(intervals) - max_non_overlapping(intervals)


def min_meeting_rooms(intervals: list[Interval]) -> int:
    """Rooms needed so no two meetings share a room = max simultaneous meetings.

    Process meetings by start time; a min-heap holds the end times of rooms in
    use. If the earliest-ending room is free by now, reuse it.
    """
    ends: list[int] = []
    for start, end in sorted(intervals):
        if ends and ends[0] <= start:
            heapq.heapreplace(ends, end)  # reuse that room
        else:
            heapq.heappush(ends, end)  # open a new room
    return len(ends)


def min_arrows(balloons: list[Interval]) -> int:
    """Fewest points such that every (closed) interval contains at least one point.

    Sort by end; shoot at the end of the first un-hit interval, which also
    hits every interval that starts at or before that point.
    """
    arrows, last_shot = 0, float("-inf")
    for start, end in sorted(balloons, key=lambda iv: iv[1]):
        if start > last_shot:
            arrows += 1
            last_shot = end
    return arrows


def can_jump(nums: list[int]) -> bool:
    """nums[i] = max jump length from i. Can we reach the last index?

    Greedy: track the farthest index reachable so far.
    """
    farthest = 0
    for i, step in enumerate(nums):
        if i > farthest:
            return False  # stuck before reaching i
        farthest = max(farthest, i + step)
    return True


def min_jumps(nums: list[int]) -> int:
    """Fewest jumps to reach the last index (-1 if impossible).

    Implicit BFS: [.., current_end] is everything reachable with `jumps`
    jumps; `farthest` is everything reachable with one more.
    """
    jumps = current_end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == current_end:  # used up this "level": must jump again
            if farthest == i:
                return -1
            jumps += 1
            current_end = farthest
    return jumps


def gas_station(gas: list[int], cost: list[int]) -> int:
    """Start index to drive once around the circuit, or -1.

    If total gas < total cost it is impossible. Otherwise, whenever the tank
    goes negative at station i, no station in [start, i] can be the answer
    (starting later within that stretch only has less fuel), so restart at i + 1.
    """
    if sum(gas) < sum(cost):
        return -1
    start = tank = 0
    for i in range(len(gas)):
        tank += gas[i] - cost[i]
        if tank < 0:
            start, tank = i + 1, 0
    return start


def fractional_knapsack(weights: list[float], values: list[float], capacity: float) -> float:
    """Items can be split: take the best value-per-weight first. O(n log n)."""
    total = 0.0
    for w, v in sorted(zip(weights, values), key=lambda wv: wv[1] / wv[0], reverse=True):
        take = min(w, capacity)
        total += v * take / w
        capacity -= take
        if capacity <= 0:
            break
    return total
