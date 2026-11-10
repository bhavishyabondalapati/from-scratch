# LRU cache

## Intuition
A cache has limited room. When it is full, throw out the entry that has gone the longest without
being touched ("least recently used"), on the bet that recently used things will be used again.

We need two things in O(1): *find* an entry by key, and *reorder* entries by recency. No single
structure does both, so combine two:
- a **hash map** from key to node (fast lookup),
- a **doubly linked list** ordered by recency (fast move-to-front and remove-from-back).

## Core idea
- `get(k)`: look up the node, unlink it, push it to the front.
- `put(k, v)`: if present, update and move to front. Otherwise, if full, unlink the node before
  the tail and delete its key from the map; then insert the new node at the front.
- Sentinel head/tail nodes mean every real node always has both neighbours, so unlink is two
  pointer assignments with no `if`.

## Complexity
`get`, `put`: O(1) time. Space O(capacity).

## Common pitfalls
- Not storing the **key** inside the node: when evicting from the list you must also delete from
  the map, and you only have the node.
- Forgetting that `get` changes recency.
- Updating an existing key but not moving it to the front.
- Singly linked list: cannot unlink a middle node in O(1) without its predecessor.
- Not thread-safe: real caches need a lock around the map + list update.

## When to use vs. alternatives
- LRU: workloads with temporal locality (recent ⇒ likely again). Simple and the common default.
- LFU: some items are popular over the long run and should survive bursts of one-off keys.
- FIFO / random eviction: cheaper bookkeeping, worse hit rate.
- In Python practice: `functools.lru_cache` or `OrderedDict.move_to_end`.

## Interview questions
1. **Why a doubly linked list and not a singly linked one or an array?** Removing an arbitrary
   node in O(1) requires access to its predecessor; arrays need O(n) shifting.
2. **Why must nodes store keys?** Eviction starts from the list tail; you need the key to remove
   the map entry.
3. **How would you make it thread-safe?** A mutex around get/put; or sharding into several
   independently locked LRUs to reduce contention.
4. **How does `OrderedDict` implement this?** The same way: hash map plus a doubly linked list.
5. **How would you add TTL (expiry)?** Store an expiry timestamp in each node; check lazily on
   `get` and/or sweep periodically.
