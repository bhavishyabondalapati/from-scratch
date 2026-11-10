# Stack & monotonic stack

## Intuition
A **stack** is "last in, first out": the most recent unfinished thing is the first to be
finished. That is exactly the shape of nested structures (brackets, function calls, undo).

A **monotonic stack** keeps its contents sorted (always increasing or always decreasing) by
popping anything that would break the order before pushing. The items on the stack are
"still waiting for an answer"; the moment a new element pops one, that new element *is* its
answer (next greater / next smaller element).

## Core idea
For next-greater: the stack holds indices with **decreasing** values. When `x` arrives, every
index on top with a smaller value has just met its first greater element. Each index is pushed
once and popped at most once → O(n) total even though there is a `while` inside the `for`.

Histogram: an **increasing** stack. When bar `i` is shorter than the top, the popped bar's
rectangle extends from just after the new top to just before `i`.

## Complexity
| Operation | Time | Space |
|---|---|---|
| push / pop / peek / get_min | O(1) | O(n) total |
| valid parentheses, RPN | O(n) | O(n) |
| next greater, daily temperatures, largest rectangle | O(n) amortized | O(n) |

## Common pitfalls
- Storing values instead of **indices** (you usually need distances or to write into `out[i]`).
- `<` vs `<=` when popping: decides how ties/duplicates are handled.
- Forgetting to flush the stack at the end (the sentinel `0` in the histogram does this).
- RPN operand order: the first pop is the *right* operand. Python `//` floors; C-style division truncates.
- Popping from an empty stack on input like `")"`.

## When to use vs. alternatives
- Matching/nesting, expression evaluation, DFS without recursion → plain stack.
- "Next/previous greater/smaller", spans, histogram areas → monotonic stack.
- Sliding-window max/min → monotonic **deque** (needs removal from the front too).

## Interview questions
1. **Why is the monotonic stack O(n) despite the nested loop?** Each element is pushed and popped
   at most once, so total pops ≤ n.
2. **How does MinStack give O(1) min?** Each entry stores the minimum of the stack at the time
   it was pushed; popping restores the previous minimum for free.
3. **Increasing or decreasing stack for "next greater"?** Decreasing: smaller elements wait on
   the stack until something larger pops them.
4. **In the histogram, what are a popped bar's left and right limits?** Right limit is the
   current index (first shorter bar to the right); left limit is the new stack top (first
   shorter bar to the left).
5. **Stack via list vs linked list?** Both O(1); a dynamic array has better cache behaviour and
   amortized O(1) push.
