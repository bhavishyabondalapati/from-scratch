# Pattern: linked list (dummy head, slow/fast, in-place reversal)

**Signal:** "reverse", "middle", "cycle", "k-th from the end", "merge lists", O(1) extra space.

## Representative LeetCode problems (names only)
- Reverse Linked List
- Linked List Cycle II
- Merge Two Sorted Lists
- Remove Nth Node From End of List
- Reverse Nodes in k-Group

## Solved example (my own wording)
**Problem.** A playlist is stored as a singly linked list of songs. Delete the song that is `n`
places from the end, visiting the list only once.

**Thinking.** I do not know the length. Send a `lead` pointer `n` songs ahead, then move `lead`
and `trail` together. When `lead` is on the last song, `trail` is right before the one to delete.
A dummy node in front handles "delete the first song".

```python
class Song:
    def __init__(self, title, next=None):
        self.title, self.next = title, next


def drop_from_end(head, n):
    dummy = Song(None, head)
    lead = trail = dummy
    for _ in range(n):
        lead = lead.next
    while lead.next:
        lead, trail = lead.next, trail.next
    trail.next = trail.next.next
    return dummy.next


# A -> B -> C -> D with n = 2 becomes A -> B -> D
```

**Complexity.** One pass, O(n) time, O(1) space.
