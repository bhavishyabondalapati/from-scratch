# Arrays & hashing

## Intuition
A hash map trades memory for time: instead of searching for something, you compute where it
lives. Most "find a pair / group things / count things" array problems drop from O(n²) to
O(n) the moment you ask "what would I like to look up instantly?" and store exactly that.

## Core idea / math
- A hash function maps a key to an integer; `index = hash(key) % num_buckets`.
- **Collisions** (two keys, same bucket) are unavoidable. Separate chaining keeps a small
  list per bucket; open addressing probes for the next free slot.
- **Load factor** α = items / buckets. Expected chain length is α, so keeping α below a
  constant (0.75 here) keeps operations O(1) on average. Doubling on resize makes the
  resize cost O(1) *amortized* per insert (total copy work 1 + 2 + 4 + ... + n < 2n).

## Complexity
| Operation | Average | Worst |
|---|---|---|
| HashMap get / put / delete | O(1) | O(n) (all keys in one bucket) |
| two_sum, longest_consecutive, product_except_self, top_k (bucket sort) | O(n) time, O(n) space | |
| group_anagrams | O(total characters) | |

## Common pitfalls
- Using a mutable object (list) as a key: keys must be hashable and must not change.
- `longest_consecutive`: forgetting the "only start at the beginning of a run" check makes it O(n²).
- `product_except_self`: division breaks when the array contains a zero.
- `two_sum`: insert *after* checking, otherwise an element pairs with itself.
- Iterating over a dict while modifying it.

## When to use vs. alternatives
- Hash map: need fast exact-match lookup, order does not matter.
- Sorting + two pointers: O(1) extra space matters, or you need ordered output.
- Balanced tree / sorted list: you need range queries or the min/max (hash maps cannot do these).
- Array indexed by value (counting): keys are small integers.

## Interview questions
1. **Why is hash map lookup O(1) on average but O(n) worst case?** With a good hash and bounded
   load factor each bucket holds O(1) keys on average; adversarial or poor hashing can put all
   keys in one bucket.
2. **Chaining vs. open addressing?** Chaining is simple and tolerates high load; open addressing
   is cache-friendly and uses no pointers but needs tombstones on delete and degrades near full.
3. **Why is appending to a dynamic array / resizing a hash map amortized O(1)?** Doubling means
   each element is copied O(1) times on average across all inserts.
4. **How do you get top-k frequent in O(n)?** Bucket sort by frequency, since frequencies are
   bounded by n (a heap would give O(n log k)).
5. **What makes a good hash function?** Deterministic, fast, and spreads keys uniformly; equal
   objects must have equal hashes.
