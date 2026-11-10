# Pattern: prefix sums

**Signal:** many range-sum queries; "number of subarrays with sum ..." where values can be
negative; lots of range increments.

## Representative LeetCode problems (names only)
- Range Sum Query - Immutable
- Range Sum Query 2D - Immutable
- Subarray Sum Equals K
- Find Pivot Index
- Corporate Flight Bookings

## Solved example (my own wording)
**Problem.** A bank account has a list of daily changes (deposits positive, withdrawals negative).
How many stretches of consecutive days had a net change of exactly `k`?

**Thinking.** Let `bal[j]` be the balance change after `j` days. A stretch from day `i` to day `j`
nets `bal[j] - bal[i]`. I want that to equal `k`, i.e. `bal[i] = bal[j] - k`. So as I walk forward
I keep a count of every balance seen so far and, at each day, add the number of earlier days whose
balance was `bal[j] - k`.

```python
def stretches_with_net(changes, k):
    seen = {0: 1}  # balance before day 0
    balance = count = 0
    for c in changes:
        balance += c
        count += seen.get(balance - k, 0)
        seen[balance] = seen.get(balance, 0) + 1
    return count


stretches_with_net([3, -1, 2, -2, 4], 4)  # 3: [3,-1,2], [2,-2,4], [4]
```

**Complexity.** O(n) time and space.
