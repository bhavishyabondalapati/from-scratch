"""Stacks and monotonic stacks."""

from __future__ import annotations

from typing import Any


class Stack:
    """LIFO stack backed by a Python list (append/pop at the end are O(1))."""

    def __init__(self) -> None:
        self._items: list[Any] = []

    def push(self, x: Any) -> None:
        self._items.append(x)

    def pop(self) -> Any:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> Any:
        if not self._items:
            raise IndexError("peek at empty stack")
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)


class MinStack:
    """Stack that also returns its minimum in O(1).

    Each entry stores (value, minimum of everything at or below it), so the
    minimum "rewinds" automatically when we pop.
    """

    def __init__(self) -> None:
        self._items: list[tuple[int, int]] = []

    def push(self, x: int) -> None:
        current_min = min(x, self._items[-1][1]) if self._items else x
        self._items.append((x, current_min))

    def pop(self) -> int:
        return self._items.pop()[0]

    def top(self) -> int:
        return self._items[-1][0]

    def get_min(self) -> int:
        return self._items[-1][1]

    def __len__(self) -> int:
        return len(self._items)


def is_valid_parentheses(s: str) -> bool:
    """True if every bracket in s is closed by the right type in the right order."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in s:
        if ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
        else:
            stack.append(ch)
    return not stack


def eval_rpn(tokens: list[str]) -> int:
    """Evaluate Reverse Polish Notation; division truncates toward zero."""
    stack: list[int] = []
    for tok in tokens:
        if tok in {"+", "-", "*", "/"}:
            b, a = stack.pop(), stack.pop()
            if tok == "+":
                stack.append(a + b)
            elif tok == "-":
                stack.append(a - b)
            elif tok == "*":
                stack.append(a * b)
            else:
                stack.append(int(a / b))  # int() truncates toward zero, // floors
        else:
            stack.append(int(tok))
    return stack[0]


def next_greater_element(nums: list[int]) -> list[int]:
    """out[i] = the first value to the right of i that is > nums[i], else -1. O(n).

    The stack holds indices still waiting for their answer; their values are
    decreasing from bottom to top. A new value resolves every smaller waiter.
    """
    out = [-1] * len(nums)
    stack: list[int] = []
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            out[stack.pop()] = x
        stack.append(i)
    return out


def daily_temperatures(temps: list[int]) -> list[int]:
    """out[i] = days to wait after day i for a strictly warmer day (0 if never)."""
    out = [0] * len(temps)
    stack: list[int] = []
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            out[j] = i - j
        stack.append(i)
    return out


def largest_rectangle_histogram(heights: list[int]) -> int:
    """Largest rectangle under a histogram. O(n).

    Keep a stack of bars with increasing heights. When a shorter bar arrives,
    every taller bar on the stack has found its right boundary; its left
    boundary is the bar beneath it on the stack.
    """
    stack: list[int] = []  # indices, heights increasing
    best = 0
    for i, h in enumerate([*heights, 0]):  # trailing 0 flushes the stack
        while stack and heights[stack[-1]] >= h:
            height = heights[stack.pop()]
            left = stack[-1] if stack else -1  # last bar strictly shorter on the left
            best = max(best, height * (i - left - 1))
        stack.append(i)
    return best
