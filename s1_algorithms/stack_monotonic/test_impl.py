import random

import pytest

from .impl import (
    MinStack,
    Stack,
    daily_temperatures,
    eval_rpn,
    is_valid_parentheses,
    largest_rectangle_histogram,
    next_greater_element,
)


def test_stack_and_min_stack_match_list_under_random_operations():
    rnd = random.Random(0)
    stack, min_stack, ref = Stack(), MinStack(), []
    for _ in range(3000):
        if ref and rnd.random() < 0.45:
            expected = ref.pop()
            assert stack.pop() == expected
            assert min_stack.pop() == expected
        else:
            x = rnd.randint(-50, 50)
            ref.append(x)
            stack.push(x)
            min_stack.push(x)
        assert len(stack) == len(min_stack) == len(ref)
        if ref:
            assert stack.peek() == min_stack.top() == ref[-1]
            assert min_stack.get_min() == min(ref)
    with pytest.raises(IndexError):
        Stack().pop()


def test_valid_parentheses():
    for good in ["", "()", "()[]{}", "{[()]}", "(([]){})"]:
        assert is_valid_parentheses(good)
    for bad in ["(", ")", "(]", "([)]", "(()", "())"]:
        assert not is_valid_parentheses(bad)


def test_eval_rpn():
    assert eval_rpn(["2", "1", "+", "3", "*"]) == 9
    assert eval_rpn(["4", "13", "5", "/", "+"]) == 6
    assert eval_rpn(["7", "-2", "/"]) == -3  # truncates toward zero, not -4
    assert eval_rpn(["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]) == 22


def test_next_greater_and_daily_temperatures_against_brute_force():
    rnd = random.Random(1)
    for _ in range(300):
        nums = [rnd.randint(0, 9) for _ in range(rnd.randint(0, 15))]
        n = len(nums)
        expected_val = [
            next((nums[j] for j in range(i + 1, n) if nums[j] > nums[i]), -1) for i in range(n)
        ]
        expected_gap = [
            next((j - i for j in range(i + 1, n) if nums[j] > nums[i]), 0) for i in range(n)
        ]
        assert next_greater_element(nums) == expected_val
        assert daily_temperatures(nums) == expected_gap


def test_largest_rectangle_against_brute_force():
    assert largest_rectangle_histogram([2, 1, 5, 6, 2, 3]) == 10
    rnd = random.Random(2)
    for _ in range(300):
        h = [rnd.randint(0, 9) for _ in range(rnd.randint(0, 12))]
        expected = max(
            (min(h[i : j + 1]) * (j - i + 1) for i in range(len(h)) for j in range(i, len(h))),
            default=0,
        )
        assert largest_rectangle_histogram(h) == expected
