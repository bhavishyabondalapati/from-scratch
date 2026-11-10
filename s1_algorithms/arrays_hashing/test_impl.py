import math
import random
from collections import Counter

from .impl import (
    HashMap,
    group_anagrams,
    longest_consecutive,
    product_except_self,
    top_k_frequent,
    two_sum,
)


def test_hashmap_matches_dict_under_random_operations():
    rnd = random.Random(0)
    ours, ref = HashMap(capacity=2), {}
    for _ in range(5000):
        key, op = rnd.randint(0, 200), rnd.choice(["put", "get", "delete"])
        if op == "put":
            value = rnd.random()
            ours.put(key, value)
            ref[key] = value
        elif op == "get":
            assert ours.get(key) == ref.get(key)
        else:
            assert ours.delete(key) == (ref.pop(key, None) is not None)
        assert len(ours) == len(ref)
        assert (key in ours) == (key in ref)
    assert dict(ours.items()) == ref


def test_hashmap_handles_colliding_keys():
    m = HashMap(capacity=1)  # everything collides at first
    for k in ["a", "b", "c", ("t", 1), 3.5]:
        m.put(k, repr(k))
    assert m.get(("t", 1)) == "('t', 1)" and m.get("missing", 7) == 7


def test_two_sum_against_brute_force():
    rnd = random.Random(1)
    for _ in range(300):
        nums = [rnd.randint(-20, 20) for _ in range(rnd.randint(0, 12))]
        target = rnd.randint(-30, 30)
        exists = any(
            nums[i] + nums[j] == target for i in range(len(nums)) for j in range(i + 1, len(nums))
        )
        got = two_sum(nums, target)
        assert (got is not None) == exists
        if got:
            i, j = got
            assert i < j and nums[i] + nums[j] == target


def test_group_anagrams():
    rnd = random.Random(2)
    words = ["".join(rnd.choice("abc") for _ in range(rnd.randint(1, 4))) for _ in range(200)]
    groups = group_anagrams(words)
    assert sorted(w for g in groups for w in g) == sorted(words)  # nothing lost
    keys = ["".join(sorted(g[0])) for g in groups]
    assert len(set(keys)) == len(keys)  # one group per anagram class
    for g in groups:
        assert len({"".join(sorted(w)) for w in g}) == 1


def test_top_k_frequent_matches_counter():
    rnd = random.Random(3)
    for _ in range(200):
        nums = [rnd.randint(0, 8) for _ in range(rnd.randint(1, 40))]
        counts = Counter(nums)
        k = rnd.randint(1, len(counts))
        got = top_k_frequent(nums, k)
        assert len(got) == k == len(set(got))
        # Ties can be broken either way, so compare the frequencies not the values.
        assert sorted(counts[x] for x in got) == sorted(c for _, c in counts.most_common(k))


def test_longest_consecutive_against_brute_force():
    rnd = random.Random(4)
    for _ in range(300):
        nums = [rnd.randint(-15, 15) for _ in range(rnd.randint(0, 20))]
        s, best = set(nums), 0
        for x in s:
            n = 0
            while x + n in s:
                n += 1
            best = max(best, n)
        assert longest_consecutive(nums) == best


def test_product_except_self_handles_zeros():
    rnd = random.Random(5)
    for _ in range(300):
        nums = [rnd.randint(-3, 3) for _ in range(rnd.randint(1, 8))]
        expected = [math.prod(nums[:i] + nums[i + 1 :]) for i in range(len(nums))]
        assert product_except_self(nums) == expected
