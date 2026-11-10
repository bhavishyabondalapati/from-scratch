# Trees: traversals, BST, LCA, serialization

## Intuition
A tree problem is almost always: "solve it for the left subtree, solve it for the right subtree,
combine at the node". The traversal order just says *when* you handle the node:
- **preorder** (node, left, right): top-down work, copying/serializing a tree;
- **inorder** (left, node, right): sorted order for a BST;
- **postorder** (left, right, node): bottom-up work, e.g. heights, deleting;
- **level order** (BFS): shortest depth, per-level answers.

A **BST** keeps `left < node < right` everywhere, so each comparison discards a whole subtree —
binary search in tree form.

## Core idea
- Iterative traversal = the recursion's call stack made explicit.
- BST delete with two children: replace the value with the inorder successor (min of the right
  subtree), then delete the successor, which has at most one child.
- LCA (general): the first node where `p` and `q` end up in different subtrees. LCA (BST): the
  first node whose value lies between `p` and `q`.
- Serialization: preorder alone is ambiguous; preorder **with null markers** is not.

## Complexity
| Operation | Time | Space |
|---|---|---|
| Any traversal | O(n) | O(h) stack (O(w) queue for BFS) |
| BST search / insert / delete | O(h): O(log n) balanced, O(n) degenerate | O(h) recursive |
| LCA general / BST | O(n) / O(h) | O(h) / O(1) |
| serialize / deserialize | O(n) | O(n) |

`h` = height, `w` = max width. The recursive traversals here build lists with `+`, which is
simple but O(n·h) copying; the iterative versions are the efficient ones.

## Common pitfalls
- Validating a BST by comparing only with direct children (must respect bounds from *all* ancestors).
- Inserting sorted data into a plain BST gives a linked list (height n). Real libraries
  self-balance (red-black, AVL).
- Deep recursion in Python (~1000 frames): prefer iterative on skewed trees.
- Forgetting to reattach the returned subtree: `root.left = bst_delete(root.left, v)`.
- Level order: reading `len(queue)` *inside* the loop condition instead of once per level.

## When to use vs. alternatives
- BST: ordered data with inserts/deletes plus range/rank queries.
- Hash map: only exact lookups (faster, unordered).
- Heap: only need min/max.
- Many LCA queries on one tree: binary lifting or Euler tour + sparse table.

## Interview questions
1. **Why does inorder traversal of a BST give sorted output?** By the invariant, everything left
   is smaller and visited first, everything right is larger and visited after.
2. **How do you delete a node with two children?** Swap in its inorder successor (or
   predecessor) and delete that node instead.
3. **Can you rebuild a tree from preorder only?** Not in general; you need null markers, or a
   second traversal (inorder), or the BST property.
4. **What is the worst-case height of a BST and how is it avoided?** n (sorted inserts);
   self-balancing rotations keep it O(log n).
5. **Explain the recursive LCA.** Each call returns p, q, or None from its subtree; a node that
   receives non-None from both sides is the split point.
