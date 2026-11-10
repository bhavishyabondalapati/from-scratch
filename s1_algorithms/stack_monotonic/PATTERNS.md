# Pattern: stack / monotonic stack

**Signal:** nested brackets, "evaluate expression", "next greater/smaller element", "how many
days until ...", "largest rectangle".

## Representative LeetCode problems (names only)
- Valid Parentheses
- Min Stack
- Evaluate Reverse Polish Notation
- Daily Temperatures
- Largest Rectangle in Histogram

## Solved example (my own wording)
**Problem.** A row of people of different heights all look to the right. For each person, report
the height of the first person to their right who is taller, or -1 if nobody is.

**Thinking.** Walk left to right keeping a stack of people who have not yet found someone taller.
Their heights on the stack are decreasing (if a taller person stood behind a shorter one in the
stack, the shorter one would already have been answered). When a new person arrives, they are the
answer for everyone on top of the stack who is shorter.

```python
def first_taller_to_right(heights):
    answer = [-1] * len(heights)
    waiting = []  # indices, heights decreasing
    for i, h in enumerate(heights):
        while waiting and heights[waiting[-1]] < h:
            answer[waiting.pop()] = h
        waiting.append(i)
    return answer


first_taller_to_right([170, 160, 165, 180, 175])  # [180, 165, 180, -1, -1]
```

**Complexity.** Everyone is pushed once and popped at most once: O(n) time, O(n) space.
