# Pattern: sliding window

**Signal:** "longest / shortest / count of **contiguous** subarray or substring such that ...".

## Representative LeetCode problems (names only)
- Longest Substring Without Repeating Characters
- Minimum Window Substring
- Longest Repeating Character Replacement
- Minimum Size Subarray Sum
- Find All Anagrams in a String

## Solved example (my own wording)
**Problem.** A fitness tracker records the calories burned each day (all positive). What is the
smallest number of consecutive days whose total is at least `goal`?

**Thinking.** Extend the window to the right until the total reaches the goal. Then the window is
valid, so try to make it shorter from the left while it stays valid. All numbers are positive, so
removing a day always lowers the total and I never need to move the left edge backward.

```python
def fewest_days(calories, goal):
    left = total = 0
    best = len(calories) + 1
    for right, c in enumerate(calories):
        total += c
        while total >= goal:
            best = min(best, right - left + 1)
            total -= calories[left]
            left += 1
    return 0 if best > len(calories) else best

fewest_days([200, 300, 100, 400, 350], 700)   # 2  (400 + 350)
```

**Complexity.** Each index enters and leaves the window once: O(n) time, O(1) space.
