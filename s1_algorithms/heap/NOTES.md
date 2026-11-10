# Heap / priority queue

## Intuition
A heap answers one question fast: "what is the smallest (or largest) item right now?" It is a
binary tree where every parent is no bigger than its children, so the minimum sits at the root.
It is **not** fully sorted; it does just enough work to keep the best item on top.

The tree is *complete* (filled level by level), so it packs into a plain array with no pointers:
`parent(i) = (i-1)//2`, `children(i) = 2i+1, 2i+2`.

## Core idea
- **push:** append at the end, then *sift up* (swap with the parent while smaller).
- **pop:** take the root, move the last leaf to the root, then *sift down* (swap with the
  smaller child while larger).
- **heapify:** sift down every internal node from the bottom up. Cost is
  Σ (nodes at height h) × h = Σ n/2^(h+1) × h = O(n).

## Complexity
| Operation | Time |
|---|---|
| peek | O(1) |
| push, pop | O(log n) |
| heapify | O(n) |
| top-k (size-k heap) | O(n log k), O(k) space |
| merge k sorted lists (N items total) | O(N log k) |
| median of stream: add / median | O(log n) / O(1) |

## Common pitfalls
- Top-k **largest** needs a **min**-heap of size k (the root is the one to beat), not a max-heap.
- Assuming the array is sorted: only the root is guaranteed.
- Python's `heapq` is min-only; negate values (or use a key) for a max-heap.
- Putting uncomparable objects in tuples: add a tie-breaker index, e.g. `(priority, i, obj)`.
- Median finder: forgetting to rebalance sizes, or letting an item sit in the wrong half.

## When to use vs. alternatives
- Repeated "extract best" with inserts in between (Dijkstra, scheduling, k-way merge) → heap.
- Need the k-th element once → quickselect is O(n) average.
- Need sorted order of everything → just sort.
- Need to delete/update arbitrary items or query ranks → balanced BST / indexed heap.

## Interview questions
1. **Why is heapify O(n) and not O(n log n)?** Half the nodes are leaves (0 work), a quarter move
   at most 1 level, and so on; the series sums to O(n).
2. **How do you find the k largest of a stream of n items?** Min-heap of size k: O(n log k) time,
   O(k) memory, works online.
3. **How does the two-heap median work?** Max-heap for the lower half, min-heap for the upper
   half, sizes kept within one; the median is at the tops.
4. **Heap vs sorted array vs BST for a priority queue?** Sorted array: O(n) insert. BST:
   O(log n) everything but more overhead. Heap: O(log n) insert/pop, O(1) peek, tiny constant.
5. **Is heap sort stable? In place?** In place and O(n log n) worst case, but not stable.
