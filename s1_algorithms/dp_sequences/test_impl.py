import random
from functools import cache
from itertools import combinations

from .impl import (
    edit_distance,
    edit_operations,
    lcs_length,
    lcs_string,
    lis_length,
    lis_quadratic,
    lis_sequence,
    longest_palindromic_subsequence,
)


def is_subsequence(small, big):
    it = iter(big)
    return all(x in it for x in small)


def test_lis_against_brute_force():
    rnd = random.Random(0)
    for _ in range(300):
        nums = [rnd.randint(0, 9) for _ in range(rnd.randint(0, 10))]
        brute = max(
            (
                k
                for k in range(len(nums) + 1)
                for c in combinations(nums, k)
                if all(a < b for a, b in zip(c, c[1:]))
            ),
            default=0,
        )
        assert lis_length(nums) == lis_quadratic(nums) == brute
        seq = lis_sequence(nums)
        assert len(seq) == brute
        assert all(a < b for a, b in zip(seq, seq[1:]))
        assert is_subsequence(seq, nums)


def test_lis_fast_matches_quadratic_on_larger_inputs():
    rnd = random.Random(1)
    for _ in range(20):
        nums = [rnd.randint(-1000, 1000) for _ in range(300)]
        assert lis_length(nums) == lis_quadratic(nums) == len(lis_sequence(nums))
    assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert lis_length([7, 7, 7]) == 1  # strictly increasing


def test_lcs_against_recursive_reference():
    rnd = random.Random(2)

    @cache
    def ref(a, b):
        if not a or not b:
            return 0
        if a[-1] == b[-1]:
            return 1 + ref(a[:-1], b[:-1])
        return max(ref(a[:-1], b), ref(a, b[:-1]))

    for _ in range(300):
        a = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 9)))
        b = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 9)))
        assert lcs_length(a, b) == ref(a, b)
        common = lcs_string(a, b)
        assert len(common) == ref(a, b)
        assert is_subsequence(common, a) and is_subsequence(common, b)
    assert lcs_string("ABCBDAB", "BDCABA") in {"BCBA", "BDAB", "BCAB"}


def apply_ops(a, ops):
    """Replay the edit script on a to check it really produces b."""
    chars, shift = list(a), 0
    for op in ops:
        kind, rest = op.split(" ", 1)
        what, pos = rest.rsplit(" at ", 1)
        pos = int(pos) + shift
        if kind == "substitute":
            chars[pos] = what.split("->")[1]
        elif kind == "delete":
            del chars[pos]
            shift -= 1
        else:
            chars.insert(pos, what)
            shift += 1
    return "".join(chars)


def test_edit_distance_against_recursive_reference():
    rnd = random.Random(3)

    @cache
    def ref(a, b):
        if not a or not b:
            return len(a) + len(b)
        if a[0] == b[0]:
            return ref(a[1:], b[1:])
        return 1 + min(ref(a[1:], b), ref(a, b[1:]), ref(a[1:], b[1:]))

    for _ in range(400):
        a = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 8)))
        b = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 8)))
        d = edit_distance(a, b)
        assert d == ref(a, b)
        assert d == edit_distance(b, a)  # symmetric
        ops = edit_operations(a, b)
        assert len(ops) == d and apply_ops(a, ops) == b
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("same", "same") == 0


def test_edit_distance_triangle_inequality():
    rnd = random.Random(4)
    for _ in range(200):
        a, b, c = ("".join(rnd.choice("ab") for _ in range(rnd.randint(0, 6))) for _ in range(3))
        assert edit_distance(a, c) <= edit_distance(a, b) + edit_distance(b, c)


def test_longest_palindromic_subsequence_against_brute_force():
    rnd = random.Random(5)
    for _ in range(200):
        s = "".join(rnd.choice("abc") for _ in range(rnd.randint(0, 10)))
        brute = max(k for k in range(len(s) + 1) for c in combinations(s, k) if c == c[::-1])
        assert longest_palindromic_subsequence(s) == brute
