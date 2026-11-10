# Pattern: sequence DP (LIS / LCS / edit distance)

**Signal:** "longest subsequence", "minimum operations to convert", two strings compared
position by position, "delete the fewest elements so that ...".

## Representative LeetCode problems (names only)
- Longest Increasing Subsequence
- Longest Common Subsequence
- Edit Distance
- Russian Doll Envelopes
- Delete Operation for Two Strings

## Solved example (my own wording)
**Problem.** A search box should suggest the dictionary word closest to what the user typed, where
"closest" means the fewest single-letter fixes (add a letter, drop a letter, or change a letter).

**Thinking.** Compare prefixes. Let `D[i][j]` be the number of fixes to turn the first `i` typed
letters into the first `j` letters of the word. If the last letters agree, nothing new is needed.
Otherwise one fix is needed, and the cheapest of "drop", "add", "change" decides which smaller
problem remains.

```python
def fixes(typed, word):
    prev = list(range(len(word) + 1))
    for i in range(1, len(typed) + 1):
        cur = [i] + [0] * len(word)
        for j in range(1, len(word) + 1):
            if typed[i - 1] == word[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[-1]


def suggest(typed, dictionary):
    return min(dictionary, key=lambda w: fixes(typed, w))


suggest("recieve", ["receipt", "receive", "deceive"])  # 'receive' (2 fixes)
```

**Complexity.** O(len(typed) × len(word)) per word.
