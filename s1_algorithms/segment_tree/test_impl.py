import math
import random

import pytest

from .impl import LazySegmentTree, SegmentTree

OPS = {
    "sum": (lambda a, b: a + b, 0, sum),
    "min": (min, math.inf, lambda xs: min(xs, default=math.inf)),
    "max": (max, -math.inf, lambda xs: max(xs, default=-math.inf)),
    "gcd": (math.gcd, 0, lambda xs: math.gcd(*xs)),
}


@pytest.mark.parametrize("name", OPS)
def test_segment_tree_matches_brute_force(name):
    op, identity, brute = OPS[name]
    rnd = random.Random(0)
    for n in [1, 2, 3, 7, 8, 13]:  # includes non-powers of two
        values = [rnd.randint(1, 60) for _ in range(n)]
        tree = SegmentTree(values, op, identity)
        for _ in range(400):
            if rnd.random() < 0.4:
                i, v = rnd.randrange(n), rnd.randint(1, 60)
                values[i] = v
                tree.update(i, v)
            lo = rnd.randint(0, n)
            hi = rnd.randint(lo, n)
            assert tree.query(lo, hi) == brute(values[lo:hi])


def test_non_commutative_operation_keeps_order():
    # String concatenation is associative but not commutative.
    letters = list("segmenttree")
    tree = SegmentTree(letters, lambda a, b: a + b, "")
    for lo in range(len(letters) + 1):
        for hi in range(lo, len(letters) + 1):
            assert tree.query(lo, hi) == "".join(letters[lo:hi])


def test_lazy_segment_tree_matches_brute_force():
    rnd = random.Random(1)
    for n in [1, 2, 5, 16, 23]:
        values = [rnd.randint(-9, 9) for _ in range(n)]
        tree = LazySegmentTree(values)
        for _ in range(600):
            lo = rnd.randint(0, n)
            hi = rnd.randint(lo, n)
            if rnd.random() < 0.5:
                v = rnd.randint(-5, 5)
                tree.range_add(lo, hi, v)
                for i in range(lo, hi):
                    values[i] += v
            else:
                assert tree.range_sum(lo, hi) == sum(values[lo:hi])
    assert LazySegmentTree([]).range_sum(0, 0) == 0
