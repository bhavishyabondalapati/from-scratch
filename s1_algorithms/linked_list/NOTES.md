# Linked list operations

## Intuition
A linked list is a chain of nodes, each holding a value and a pointer to the next node. You
cannot jump to position `i`; you can only walk. In exchange, inserting or removing a node is just
re-pointing one or two links: O(1) once you are standing next to it.

Three tools solve almost every linked-list problem:
1. **Dummy head:** a fake node before the real head so "delete/insert at the head" is not a special case.
2. **Slow/fast pointers:** find the middle, detect a cycle, find the k-th from the end.
3. **In-place reversal:** walk the list flipping each `next` pointer backward.

## Core math (Floyd cycle detection)
Let `a` = steps from head to the cycle start, `c` = cycle length. When slow (1 step) and fast
(2 steps) meet, fast has walked exactly twice as far, and their difference is a multiple of `c`.
So `(a + b) ≡ 0 (mod c)` where `b` is the meeting offset inside the cycle. Starting one pointer
at head and one at the meeting point and stepping both by 1, they meet after `a` steps at the
cycle start.

## Complexity
| Operation | Time | Extra space |
|---|---|---|
| reverse, middle, cycle, merge, remove n-th, reverse k-group | O(n) | O(1) (recursive reverse: O(n) stack) |
| sort_list (merge sort) | O(n log n) | O(log n) stack |
| access by index | O(n) | |

## Common pitfalls
- Losing the rest of the list: save `cur.next` before overwriting it.
- Forgetting the dummy node, then crashing when the head itself is removed.
- `while fast and fast.next` — checking only `fast.next` crashes on an empty list.
- In `sort_list`, splitting at the second middle makes a 2-node list recurse forever.
- Comparing nodes with `==` on values instead of identity (`is`) in cycle problems.

## When to use vs. alternatives
- Linked list: O(1) insert/delete given the node (LRU cache), no reallocation, splice lists.
- Array: random access, cache-friendly, binary search. Usually faster in practice.
- Sorting: merge sort suits linked lists (no random access needed); quicksort suits arrays.

## Interview questions
1. **Reverse a linked list iteratively and recursively; space?** Iterative O(1); recursive O(n)
   call stack.
2. **Why do slow/fast pointers detect a cycle?** Inside the cycle the gap between them shrinks by
   one each step, so fast must land on slow.
3. **How do you find where the cycle starts?** After they meet, reset one pointer to the head and
   advance both one step at a time; they meet at the entry (see math above).
4. **Why use a dummy head?** It gives every real node a predecessor, removing head special cases.
5. **Why merge sort rather than quicksort for linked lists?** Merging needs only sequential
   access and O(1) extra space; quicksort's partition relies on random access.
