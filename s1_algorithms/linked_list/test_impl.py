import random

from .impl import (
    cycle_start,
    from_list,
    has_cycle,
    merge_sorted,
    middle,
    remove_nth_from_end,
    reverse,
    reverse_k_group,
    reverse_recursive,
    sort_list,
    to_list,
)


def rand_list(rnd, lo=0, hi=12):
    return [rnd.randint(-20, 20) for _ in range(rnd.randint(lo, hi))]


def test_round_trip_and_reverse_match_python_lists():
    rnd = random.Random(0)
    for _ in range(300):
        xs = rand_list(rnd)
        assert to_list(from_list(xs)) == xs
        assert to_list(reverse(from_list(xs))) == xs[::-1]
        assert to_list(reverse_recursive(from_list(xs))) == xs[::-1]


def test_middle():
    for n in range(1, 12):
        xs = list(range(n))
        assert middle(from_list(xs)).val == xs[n // 2]
    assert middle(None) is None


def test_cycle_detection_and_start():
    rnd = random.Random(1)
    for _ in range(200):
        n = rnd.randint(1, 15)
        head = from_list(range(n))
        nodes = []
        cur = head
        while cur:
            nodes.append(cur)
            cur = cur.next
        assert not has_cycle(head) and cycle_start(head) is None
        entry = rnd.randrange(n)
        nodes[-1].next = nodes[entry]  # tail points back into the list
        assert has_cycle(head)
        assert cycle_start(head) is nodes[entry]


def test_merge_sorted_and_sort_list():
    rnd = random.Random(2)
    for _ in range(300):
        a, b = sorted(rand_list(rnd)), sorted(rand_list(rnd))
        assert to_list(merge_sorted(from_list(a), from_list(b))) == sorted(a + b)
        xs = rand_list(rnd, hi=30)
        assert to_list(sort_list(from_list(xs))) == sorted(xs)


def test_remove_nth_from_end():
    rnd = random.Random(3)
    for _ in range(300):
        xs = rand_list(rnd, lo=1)
        n = rnd.randint(1, len(xs))
        expected = xs[: len(xs) - n] + xs[len(xs) - n + 1 :]
        assert to_list(remove_nth_from_end(from_list(xs), n)) == expected


def test_reverse_k_group():
    rnd = random.Random(4)
    for _ in range(300):
        xs = rand_list(rnd)
        k = rnd.randint(1, 5)
        expected = []
        for i in range(0, len(xs), k):
            chunk = xs[i : i + k]
            expected += chunk[::-1] if len(chunk) == k else chunk
        assert to_list(reverse_k_group(from_list(xs), k)) == expected
