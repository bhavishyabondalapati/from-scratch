"""Binary trees: traversals (recursive + iterative), BST operations, LCA, serialization."""

from __future__ import annotations

from collections import deque


class TreeNode:
    def __init__(
        self, val: int, left: TreeNode | None = None, right: TreeNode | None = None
    ) -> None:
        self.val, self.left, self.right = val, left, right


# --- Traversals --------------------------------------------------------------


def inorder_recursive(root: TreeNode | None) -> list[int]:
    """Left, node, right. On a BST this visits values in sorted order."""
    if root is None:
        return []
    return inorder_recursive(root.left) + [root.val] + inorder_recursive(root.right)


def preorder_recursive(root: TreeNode | None) -> list[int]:
    if root is None:
        return []
    return [root.val] + preorder_recursive(root.left) + preorder_recursive(root.right)


def postorder_recursive(root: TreeNode | None) -> list[int]:
    if root is None:
        return []
    return postorder_recursive(root.left) + postorder_recursive(root.right) + [root.val]


def inorder_iterative(root: TreeNode | None) -> list[int]:
    """Simulate the call stack: go left as far as possible, visit, then go right."""
    out: list[int] = []
    stack: list[TreeNode] = []
    node = root
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.val)
        node = node.right
    return out


def preorder_iterative(root: TreeNode | None) -> list[int]:
    out: list[int] = []
    stack = [root] if root else []
    while stack:
        node = stack.pop()
        out.append(node.val)
        if node.right:  # push right first so left is popped (visited) first
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return out


def postorder_iterative(root: TreeNode | None) -> list[int]:
    """Each node goes on the stack with a flag: has its subtree been expanded yet?"""
    out: list[int] = []
    stack: list[tuple[TreeNode, bool]] = [(root, False)] if root else []
    while stack:
        node, expanded = stack.pop()
        if expanded:
            out.append(node.val)
            continue
        stack.append((node, True))  # come back to it after both children
        if node.right:
            stack.append((node.right, False))
        if node.left:
            stack.append((node.left, False))
    return out


def level_order(root: TreeNode | None) -> list[list[int]]:
    """Breadth-first: one list of values per depth."""
    levels: list[list[int]] = []
    queue = deque([root] if root else [])
    while queue:
        level = []
        for _ in range(len(queue)):  # exactly the nodes of the current depth
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        levels.append(level)
    return levels


def height(root: TreeNode | None) -> int:
    """Number of nodes on the longest root-to-leaf path (empty tree = 0)."""
    return 0 if root is None else 1 + max(height(root.left), height(root.right))


# --- Binary search tree --------------------------------------------------------
# Invariant: everything in the left subtree < node.val < everything in the right.


def bst_insert(root: TreeNode | None, val: int) -> TreeNode:
    """Insert val (duplicates ignored); returns the (possibly new) root."""
    if root is None:
        return TreeNode(val)
    if val < root.val:
        root.left = bst_insert(root.left, val)
    elif val > root.val:
        root.right = bst_insert(root.right, val)
    return root


def bst_search(root: TreeNode | None, val: int) -> TreeNode | None:
    while root and root.val != val:
        root = root.left if val < root.val else root.right
    return root


def bst_min(root: TreeNode) -> TreeNode:
    while root.left:
        root = root.left
    return root


def bst_delete(root: TreeNode | None, val: int) -> TreeNode | None:
    """Delete val if present; returns the new root.

    Three cases for the node being removed:
    - no children: just drop it;
    - one child: replace it with that child;
    - two children: copy in its inorder successor (smallest value in the right
      subtree), then delete that successor from the right subtree.
    """
    if root is None:
        return None
    if val < root.val:
        root.left = bst_delete(root.left, val)
    elif val > root.val:
        root.right = bst_delete(root.right, val)
    else:
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        successor = bst_min(root.right)
        root.val = successor.val
        root.right = bst_delete(root.right, successor.val)
    return root


def is_valid_bst(root: TreeNode | None) -> bool:
    """Check the BST invariant by passing allowed (low, high) bounds downward."""

    def check(node: TreeNode | None, low: float, high: float) -> bool:
        if node is None:
            return True
        if not low < node.val < high:
            return False
        return check(node.left, low, node.val) and check(node.right, node.val, high)

    return check(root, float("-inf"), float("inf"))


def kth_smallest(root: TreeNode | None, k: int) -> int:
    """k-th smallest value (1-indexed) via an early-stopping inorder walk."""
    stack: list[TreeNode] = []
    node = root
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        k -= 1
        if k == 0:
            return node.val
        node = node.right
    raise IndexError("k is larger than the tree")


# --- Lowest common ancestor ------------------------------------------------------


def lca(root: TreeNode | None, p: TreeNode, q: TreeNode) -> TreeNode | None:
    """LCA in any binary tree (p and q are nodes assumed to be in the tree).

    Returns p or q if found in this subtree. If p and q are found in different
    subtrees of a node, that node is where their paths split: the LCA.
    """
    if root is None or root is p or root is q:
        return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    if left and right:
        return root
    return left or right


def lca_bst(root: TreeNode | None, p: int, q: int) -> TreeNode | None:
    """LCA in a BST using the ordering: O(height), no recursion."""
    while root:
        if p < root.val and q < root.val:
            root = root.left
        elif p > root.val and q > root.val:
            root = root.right
        else:
            return root  # p and q are on different sides (or one equals root)
    return None


# --- Serialize / deserialize -------------------------------------------------------


def serialize(root: TreeNode | None) -> str:
    """Preorder with '#' marking missing children, e.g. '1,2,#,#,3,#,#'.

    Recording the null markers makes preorder alone enough to rebuild the tree.
    """
    parts: list[str] = []

    def walk(node: TreeNode | None) -> None:
        if node is None:
            parts.append("#")
            return
        parts.append(str(node.val))
        walk(node.left)
        walk(node.right)

    walk(root)
    return ",".join(parts)


def deserialize(data: str) -> TreeNode | None:
    tokens = iter(data.split(","))

    def build() -> TreeNode | None:
        tok = next(tokens)
        if tok == "#":
            return None
        node = TreeNode(int(tok))
        node.left = build()  # consumes exactly the left subtree's tokens
        node.right = build()
        return node

    return build()
