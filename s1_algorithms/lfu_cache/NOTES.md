# LFU cache

## Intuition
Evict the entry that has been used the **fewest times**. If several entries tie, evict the one
used least recently among them. Compared with LRU, a key that is popular over the long run is
not pushed out by a short burst of one-off keys.

## Core idea
Bucket keys by frequency: `groups[f]` is an ordered set of the keys used exactly `f` times
(oldest first). Using a key moves it from `groups[f]` to `groups[f + 1]`. Track `min_freq` so the
victim is always "the oldest key in `groups[min_freq]`".

Why `min_freq` is O(1) to maintain:
- inserting a new key ⇒ `min_freq = 1`;
- bumping the last key out of `groups[min_freq]` ⇒ that key is now in `groups[min_freq + 1]`,
  so the new minimum is exactly `min_freq + 1`;
- eviction is always followed by an insert, which resets it to 1.

## Complexity
`get`, `put`: O(1). Space O(capacity). (A heap-based LFU would be O(log n) per operation.)

## Common pitfalls
- Forgetting the LRU tie-break inside a frequency group.
- Not updating `min_freq` when a group becomes empty.
- `put` on an existing key must also count as a use.
- Capacity 0 must not crash.
- **Cache pollution:** a key that was hot long ago keeps a high count forever. Real systems
  decay counts over time (e.g. TinyLFU, Redis's approximate LFU).

## When to use vs. alternatives
- LFU: stable popularity distribution (CDN objects, hot database rows).
- LRU: popularity shifts quickly; simpler; usually the default.
- ARC / TinyLFU / W-TinyLFU: adaptive blends used in production caches.

## Interview questions
1. **How do you get O(1) LFU?** Hash map for values and frequencies, plus one ordered set per
   frequency and a `min_freq` pointer.
2. **Why can `min_freq` just be incremented?** The key leaving the minimum group lands in the
   group one higher, which is therefore non-empty.
3. **LFU vs LRU trade-offs?** LFU protects long-term popular items but adapts slowly and suffers
   from stale high counts; LRU adapts fast but one scan of cold keys can flush hot ones.
4. **How would you fix stale counts?** Periodically halve all counts (aging) or use a time window.
5. **What would a heap-based design cost?** O(log n) per access, since each use changes a key's
   priority.
