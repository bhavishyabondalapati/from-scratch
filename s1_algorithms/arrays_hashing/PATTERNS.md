# Pattern: arrays & hashing

**Signal:** "have I seen this before?", "how many times?", "group by some signature",
"find a pair with property X" on unsorted data.

## Representative LeetCode problems (names only)
- Two Sum
- Group Anagrams
- Top K Frequent Elements
- Longest Consecutive Sequence
- Product of Array Except Self

## Solved example (my own wording)
**Problem.** A shop logs the price of each sale in order. Given the list of prices and a gift
card value, find two different sales whose prices add up to exactly the gift card value, and
return their positions.

**Thinking.** Checking every pair is O(n²). Instead, walk the list once. At each price `p`, the
partner I need is `card - p`. If I have already seen that partner, I am done. So I keep a dict
from price to the position where I saw it.

```python
def gift_card_pair(prices, card):
    seen = {}  # price -> position
    for j, p in enumerate(prices):
        if card - p in seen:
            return seen[card - p], j
        seen[p] = j  # store AFTER checking so p cannot pair with itself
    return None


gift_card_pair([4, 9, 5, 1], 10)  # (1, 3): 9 + 1
```

**Complexity.** One pass, O(n) time, O(n) extra space.
