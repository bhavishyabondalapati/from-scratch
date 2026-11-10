# Pattern: trie

**Signal:** "starts with", autocomplete, many words searched at once, wildcard dictionary lookup,
maximum XOR.

## Representative LeetCode problems (names only)
- Implement Trie (Prefix Tree)
- Design Add and Search Words Data Structure
- Word Search II
- Search Suggestions System
- Maximum XOR of Two Numbers in an Array

## Solved example (my own wording)
**Problem.** A contacts app stores names. As the user types, show how many contacts start with
what has been typed so far.

**Thinking.** Put each name into a trie and, on every node along its path, add 1 to a counter.
The counter at the node reached by the typed text is the number of names beginning with it.

```python
class Node:
    def __init__(self):
        self.kids, self.count = {}, 0


class Contacts:
    def __init__(self):
        self.root = Node()

    def add(self, name):
        node = self.root
        for ch in name:
            node = node.kids.setdefault(ch, Node())
            node.count += 1

    def matches(self, typed):
        node = self.root
        for ch in typed:
            if ch not in node.kids:
                return 0
            node = node.kids[ch]
        return node.count


c = Contacts()
for n in ["sam", "sara", "sandeep", "tom"]:
    c.add(n)
c.matches("sa")  # 3
```

**Complexity.** O(length of the name) per add and per query.
