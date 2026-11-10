import random

import pytest

from .impl import (
    TreeNode,
    bst_delete,
    bst_insert,
    bst_search,
    deserialize,
    height,
    inorder_iterative,
    inorder_recursive,
    is_valid_bst,
    kth_smallest,
    lca,
    lca_bst,
    level_order,
    postorder_iterative,
    postorder_recursive,
    preorder_iterative,
    preorder_recursive,
    serialize,
)


def random_tree(rnd, n):
    """Random-shaped binary tree with values 0..n-1; returns (root, nodes, parent map)."""
    if n == 0:
        return None, [], {}
    nodes = [TreeNode(i) for i in range(n)]
    parent = {0: None}
    free = [(0, "left"), (0, "right")]
    for i in range(1, n):
        p, side = free.pop(rnd.randrange(len(free)))
        setattr(nodes[p], side, nodes[i])
        parent[i] = p
        free += [(i, "left"), (i, "right")]
    return nodes[0], nodes, parent


def same_tree(a, b):
    if a is None or b is None:
        return a is b
    return a.val == b.val and same_tree(a.left, b.left) and same_tree(a.right, b.right)


def test_iterative_traversals_match_recursive():
    rnd = random.Random(0)
    for _ in range(300):
        root, _, _ = random_tree(rnd, rnd.randint(0, 25))
        assert inorder_iterative(root) == inorder_recursive(root)
        assert preorder_iterative(root) == preorder_recursive(root)
        assert postorder_iterative(root) == postorder_recursive(root)


def test_known_traversals_and_levels():
    #       1
    #      / \
    #     2   3
    #    / \   \
    #   4   5   6
    root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, None, TreeNode(6)))
    assert inorder_recursive(root) == [4, 2, 5, 1, 3, 6]
    assert preorder_recursive(root) == [1, 2, 4, 5, 3, 6]
    assert postorder_recursive(root) == [4, 5, 2, 6, 3, 1]
    assert level_order(root) == [[1], [2, 3], [4, 5, 6]]
    assert height(root) == 3 and height(None) == 0 and level_order(None) == []


def test_bst_behaves_like_a_sorted_set():
    rnd = random.Random(1)
    root, ref = None, set()
    for _ in range(4000):
        x = rnd.randint(0, 60)
        op = rnd.choice(["insert", "delete", "search"])
        if op == "insert":
            root = bst_insert(root, x)
            ref.add(x)
        elif op == "delete":
            root = bst_delete(root, x)
            ref.discard(x)
        else:
            assert (bst_search(root, x) is not None) == (x in ref)
        assert inorder_iterative(root) == sorted(ref)
        assert is_valid_bst(root)
    for k, v in enumerate(sorted(ref), start=1):
        assert kth_smallest(root, k) == v
    with pytest.raises(IndexError):
        kth_smallest(root, len(ref) + 1)


def test_is_valid_bst_rejects_deep_violation():
    # 6 is in the left subtree of 5 but greater than 5: locally fine, globally wrong.
    assert not is_valid_bst(TreeNode(5, TreeNode(3, None, TreeNode(6)), TreeNode(8)))
    assert not is_valid_bst(TreeNode(2, TreeNode(2)))  # duplicates not allowed
    assert is_valid_bst(None)


def test_lca_matches_parent_pointer_reference():
    rnd = random.Random(2)
    for _ in range(200):
        n = rnd.randint(1, 25)
        root, nodes, parent = random_tree(rnd, n)
        a, b = rnd.randrange(n), rnd.randrange(n)
        ancestors = set()
        x = a
        while x is not None:
            ancestors.add(x)
            x = parent[x]
        x = b
        while x not in ancestors:
            x = parent[x]
        assert lca(root, nodes[a], nodes[b]) is nodes[x]


def test_lca_bst_agrees_with_general_lca():
    rnd = random.Random(3)
    for _ in range(200):
        values = rnd.sample(range(100), rnd.randint(1, 20))
        root = None
        for v in values:
            root = bst_insert(root, v)
        p, q = rnd.choice(values), rnd.choice(values)
        assert lca_bst(root, p, q) is lca(root, bst_search(root, p), bst_search(root, q))


def test_serialize_round_trip():
    rnd = random.Random(4)
    assert serialize(None) == "#" and deserialize("#") is None
    for _ in range(300):
        root, _, _ = random_tree(rnd, rnd.randint(0, 25))
        data = serialize(root)
        rebuilt = deserialize(data)
        assert same_tree(root, rebuilt)
        assert serialize(rebuilt) == data
    assert serialize(TreeNode(-1, TreeNode(20))) == "-1,20,#,#,#"
