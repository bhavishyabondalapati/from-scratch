import random

import pytest

from .impl import LFUCache


class SlowLFU:
    """Obviously-correct O(n) reference: scan for (lowest freq, oldest use)."""

    def __init__(self, capacity):
        self.capacity, self.clock = capacity, 0
        self.data = {}  # key -> [value, freq, last_used]

    def _tick(self):
        self.clock += 1
        return self.clock

    def get(self, key):
        if key not in self.data:
            return None
        entry = self.data[key]
        entry[1] += 1
        entry[2] = self._tick()
        return entry[0]

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.data:
            entry = self.data[key]
            entry[0], entry[1], entry[2] = value, entry[1] + 1, self._tick()
            return
        if len(self.data) == self.capacity:
            victim = min(self.data, key=lambda k: (self.data[k][1], self.data[k][2]))
            del self.data[victim]
        self.data[key] = [value, 1, self._tick()]


def test_classic_example():
    c = LFUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    assert c.get(1) == 1  # freq: 1->2
    c.put(3, 3)  # evicts key 2 (freq 1)
    assert c.get(2) is None and c.get(3) == 3
    c.put(4, 4)  # 1 and 3 both have freq 2; 1 is least recent -> evicted
    assert c.get(1) is None and c.get(3) == 3 and c.get(4) == 4


@pytest.mark.parametrize("capacity", [0, 1, 2, 4, 10])
def test_matches_slow_reference(capacity):
    rnd = random.Random(capacity)
    ours, ref = LFUCache(capacity), SlowLFU(capacity)
    for step in range(6000):
        key = rnd.randint(0, 2 * capacity + 2)
        if rnd.random() < 0.5:
            assert ours.get(key) == ref.get(key)
        else:
            ours.put(key, step)
            ref.put(key, step)
        assert len(ours) == len(ref.data)
        assert set(ours._values) == set(ref.data)
