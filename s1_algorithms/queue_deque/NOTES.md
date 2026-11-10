# Queue, deque, monotonic queue

## Intuition
- **Queue:** first in, first out, like a line at a shop. Used for BFS and scheduling.
- **Deque:** a line where people can join or leave at *both* ends.
- **Monotonic queue:** a deque kept sorted by throwing out elements that can never matter again.
  For "max of every window": if a newer element is bigger than an older one, the older one will
  leave the window first *and* is smaller, so it is useless. Evict it.

## Core idea
Ring buffer: store items in a fixed array, track `head` and `size`; position of item `i` is
`(head + i) % capacity`. No shifting is ever needed, so both ends are O(1). When full, copy into
an array twice the size (amortized O(1), same argument as a dynamic array).

Two-stack queue: each element is pushed to `inbox` once, moved to `outbox` once, popped once →
at most 3 operations per element → amortized O(1).

## Complexity
| Operation | Time |
|---|---|
| Deque append / appendleft / pop / popleft / index | O(1) amortized |
| Two-stack queue push / pop | O(1) amortized (single pop can be O(n)) |
| sliding_window_max, shortest_subarray_sum_at_least | O(n) time, O(k) / O(n) space |

## Common pitfalls
- Using `list.pop(0)` as a queue: it is O(n) because every element shifts.
- Ring buffer growth: copying the raw array instead of unrolling from `head`.
- Monotonic deque storing values instead of indices (you need indices to know when the front expires).
- Evicting with `<` vs `<=`: with `<`, equal values pile up (still correct, just bigger).

## When to use vs. alternatives
- Window max/min → monotonic deque O(n); a heap also works but is O(n log n).
- Shortest subarray with sum ≥ k: positives only → sliding window; negatives allowed → prefix
  sums + monotonic deque.
- Need priority rather than arrival order → heap.

## Interview questions
1. **Implement a queue with two stacks; what is the complexity?** Amortized O(1): each element
   crosses from inbox to outbox exactly once.
2. **Why is sliding window max O(n)?** Each index enters and leaves the deque at most once.
3. **Why can smaller-and-older elements be discarded?** They expire sooner and are dominated by
   the newer larger element for every window that contains both.
4. **How does a ring buffer avoid shifting?** Head pointer plus modulo arithmetic; removing from
   the front just advances `head`.
5. **Deque vs linked list implementation?** Linked list has true O(1) worst case but poor cache
   locality and per-node overhead; ring buffer is contiguous and supports O(1) indexing.
