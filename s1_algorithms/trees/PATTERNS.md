# Pattern: tree DFS / BFS / BST

**Signal:** anything with a root and children; "depth", "path", "ancestor", "level", "validate",
"k-th smallest in a BST".

## Representative LeetCode problems (names only)
- Binary Tree Level Order Traversal
- Validate Binary Search Tree
- Kth Smallest Element in a BST
- Lowest Common Ancestor of a Binary Tree
- Serialize and Deserialize Binary Tree

## Solved example (my own wording)
**Problem.** A company org chart is a binary tree (each manager has at most two direct reports).
Given two employees, find the lowest-ranking manager who is above both of them (an employee counts
as above themselves).

**Thinking.** Search the tree. A subtree "reports" an employee upward if it contains one of the
two. If a manager hears a report from both the left and the right side, the two employees sit in
different branches below that manager, so that manager is the answer. If only one side reports,
pass that report up unchanged.

```python
class Person:
    def __init__(self, name, left=None, right=None):
        self.name, self.left, self.right = name, left, right


def common_manager(node, a, b):
    if node is None or node is a or node is b:
        return node
    left = common_manager(node.left, a, b)
    right = common_manager(node.right, a, b)
    if left and right:
        return node
    return left or right


dev1, dev2, designer = Person("dev1"), Person("dev2"), Person("designer")
lead = Person("lead", dev1, dev2)
ceo = Person("ceo", lead, designer)
common_manager(ceo, dev1, dev2).name  # 'lead'
common_manager(ceo, dev2, designer).name  # 'ceo'
```

**Complexity.** O(n) time, O(height) recursion stack.
