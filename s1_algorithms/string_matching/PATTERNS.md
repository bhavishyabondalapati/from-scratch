# Pattern: string matching (KMP / rolling hash)

**Signal:** "find occurrences of a pattern", "prefix that is also a suffix", "repeated
substring", "is s a rotation of t", comparing many substrings quickly.

## Representative LeetCode problems (names only)
- Find the Index of the First Occurrence in a String
- Repeated Substring Pattern
- Longest Happy Prefix
- Shortest Palindrome
- Longest Duplicate Substring

## Solved example (my own wording)
**Problem.** A necklace of coloured beads is described by a string read clockwise from some bead.
Two descriptions may start at different beads. Decide whether two descriptions are the same
necklace.

**Thinking.** If `t` is `s` started from a different bead, then `t` is a rotation of `s`, and every
rotation of `s` appears as a substring of `s + s`. So the question is a single pattern search,
which KMP does in linear time.

```python
def same_necklace(s, t):
    if len(s) != len(t):
        return False
    text, pattern = s + s, t
    if not pattern:
        return True
    lps, k = [0] * len(pattern), 0
    for i in range(1, len(pattern)):
        while k and pattern[i] != pattern[k]:
            k = lps[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        lps[i] = k
    k = 0
    for ch in text:
        while k and ch != pattern[k]:
            k = lps[k - 1]
        if ch == pattern[k]:
            k += 1
        if k == len(pattern):
            return True
    return False


same_necklace("rgbrb", "brbrg")  # True
same_necklace("rgbrb", "rbgrb")  # False
```

**Complexity.** O(n) time and space.
