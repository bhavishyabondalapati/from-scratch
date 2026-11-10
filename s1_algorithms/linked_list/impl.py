"""Singly linked list operations."""

from __future__ import annotations

from collections.abc import Iterable


class ListNode:
    def __init__(self, val: int = 0, next: ListNode | None = None) -> None:
        self.val = val
        self.next = next


def from_list(values: Iterable[int]) -> ListNode | None:
    """Build a linked list from a Python iterable."""
    dummy = tail = ListNode()
    for v in values:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def to_list(head: ListNode | None) -> list[int]:
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


def reverse(head: ListNode | None) -> ListNode | None:
    """Reverse iteratively: flip each `next` pointer to point backwards."""
    prev = None
    while head:
        head.next, prev, head = prev, head, head.next
    return prev


def reverse_recursive(head: ListNode | None) -> ListNode | None:
    if head is None or head.next is None:
        return head
    new_head = reverse_recursive(head.next)  # reverse everything after head
    head.next.next = head  # the old second node now points back at head
    head.next = None
    return new_head


def middle(head: ListNode | None) -> ListNode | None:
    """Middle node (second middle when the length is even), slow/fast pointers."""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow


def has_cycle(head: ListNode | None) -> bool:
    """Floyd's tortoise and hare: a fast pointer laps a slow one iff there is a cycle."""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            return True
    return False


def cycle_start(head: ListNode | None) -> ListNode | None:
    """Node where the cycle begins, or None.

    Let a = distance head -> cycle start, b = start -> meeting point, c = cycle
    length. Slow walked a + b, fast walked 2(a + b), and the difference is a
    multiple of c. So a + b = m*c, i.e. walking `a` more steps from the meeting
    point lands on the cycle start; so does walking `a` steps from head.
    """
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            slow = head
            while slow is not fast:
                slow, fast = slow.next, fast.next
            return slow
    return None


def merge_sorted(a: ListNode | None, b: ListNode | None) -> ListNode | None:
    """Merge two sorted lists by re-linking nodes (no new nodes)."""
    dummy = tail = ListNode()
    while a and b:
        if a.val <= b.val:
            tail.next, a = a, a.next
        else:
            tail.next, b = b, b.next
        tail = tail.next
    tail.next = a or b
    return dummy.next


def remove_nth_from_end(head: ListNode | None, n: int) -> ListNode | None:
    """Remove the n-th node from the end (n >= 1) in one pass.

    Move `lead` n nodes ahead, then advance both; when `lead` hits the end,
    `trail` sits just before the node to delete.
    """
    dummy = ListNode(0, head)
    lead = trail = dummy
    for _ in range(n):
        lead = lead.next
    while lead.next:
        lead, trail = lead.next, trail.next
    trail.next = trail.next.next
    return dummy.next


def reverse_k_group(head: ListNode | None, k: int) -> ListNode | None:
    """Reverse nodes in groups of k; a final group shorter than k is left as is."""
    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        kth = group_prev  # find the k-th node of this group
        for _ in range(k):
            kth = kth.next
            if kth is None:
                return dummy.next
        group_next = kth.next
        # Reverse the group, with `prev` starting at group_next so the old
        # first node ends up pointing at the rest of the list.
        prev, cur = group_next, group_prev.next
        while cur is not group_next:
            cur.next, prev, cur = prev, cur, cur.next
        new_tail = group_prev.next  # old first node is now the group's tail
        group_prev.next = kth
        group_prev = new_tail


def sort_list(head: ListNode | None) -> ListNode | None:
    """Merge sort on a linked list: O(n log n) time, O(log n) recursion stack."""
    if head is None or head.next is None:
        return head
    slow, fast = head, head.next  # fast starts one ahead so slow stops at the FIRST middle
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    second, slow.next = slow.next, None  # cut the list in two
    return merge_sorted(sort_list(head), sort_list(second))
