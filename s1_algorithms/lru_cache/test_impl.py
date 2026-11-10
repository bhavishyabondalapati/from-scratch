import random
from collections import OrderedDict

import pytest

from .impl import LRUCache


class ReferenceLRU:
    """Trusted reference built on OrderedDict."""

    def __init__(self, capacity):
        self.capacity, self.d = capacity, OrderedDict()

    def get(self, key):
        if key not in self.d:
            return None
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.capacity:
            self.d.popitem(last=False)


def test_classic_example():
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1
    c.put(3, 3)  # evicts key 2 (1 was just used)
    assert c.get(2) is None
    c.put(4, 4)  # evicts key 1
    assert c.get(1) is None and c.get(3) == 3 and c.get(4) == 4


@pytest.mark.parametrize("capacity", [1, 2, 5, 16])
def test_matches_ordered_dict_reference(capacity):
    rnd = random.Random(capacity)
    ours, ref = LRUCache(capacity), ReferenceLRU(capacity)
    for step in range(5000):
        key = rnd.randint(0, 3 * capacity)
        if rnd.random() < 0.5:
            assert ours.get(key) == ref.get(key)
        else:
            ours.put(key, step)
            ref.put(key, step)
        assert len(ours) == len(ref.d) <= capacity
        assert ours.keys_by_recency() == list(reversed(ref.d))


def test_put_existing_key_updates_value_and_recency():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)  # "a" becomes most recent
    c.put("c", 3)  # so "b" is evicted
    assert c.get("a") == 10 and c.get("b") is None


def test_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache(0)
