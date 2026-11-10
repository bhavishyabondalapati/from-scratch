# Decision tree (CART, gini + entropy)

## Intuition
A decision tree is a game of twenty questions. At each step it asks one yes/no question about one
feature ("is petal length ≤ 2.45?"), sending the data left or right. It keeps asking until the
group that remains is (mostly) one class, then predicts that class. For regression it predicts the
group's average.

How does it pick the question? It tries *every* feature and *every* threshold and keeps the one
that makes the two resulting groups as "pure" as possible. It does this greedily, one split at a
time, never looking ahead.

## Core math
- Gini impurity: `G = 1 − Σₖ pₖ²`. Entropy: `H = −Σₖ pₖ log₂ pₖ`. Both are 0 for a pure node
  and maximal for a uniform mix.
- Regression impurity: variance of the targets (so splits minimise squared error).
- Quality of a split: weighted child impurity `(n_L/n)·I(L) + (n_R/n)·I(R)`;
  **information gain** = parent impurity − that.
- CART = binary splits, threshold at the midpoint between consecutive sorted values.
- Efficient search: sort a feature once, then a running (cumulative) class count gives the
  impurity of every split position in O(n).

## Complexity
| | Time | Space |
|---|---|---|
| Train | O(d · n log n) per level → O(d · n log n · depth) | O(n) + tree |
| Predict | O(depth) per sample | |

Depth is O(log n) for a balanced tree, up to O(n) for a degenerate one.

## Common pitfalls
- **Overfitting:** an unrestricted tree memorises the training set. Limit `max_depth`,
  `min_samples_leaf`, or prune (cost-complexity pruning).
- **High variance:** small changes in data can change the top split and thus the whole tree.
  This is exactly what bagging / random forests fix.
- Greedy splitting cannot see XOR-like interactions at depth 1.
- Axis-aligned splits approximate diagonal boundaries with a staircase.
- Impurity-based feature importance is biased toward high-cardinality features.
- Trees cannot extrapolate: regression predictions are capped by training target values.

## When to use vs. alternatives
- Mixed feature types, no scaling needed, non-linear interactions, need an explainable model.
- Single trees are rarely the most accurate: use random forests (lower variance) or gradient
  boosting (lower bias) for accuracy.
- Smooth/linear relationships → linear models are more data-efficient.

## Interview questions
1. **Gini vs entropy?** Very similar in practice; gini is slightly cheaper (no log), entropy
   penalises mixed nodes a little more. They rarely change the chosen split.
2. **How does a tree handle continuous features?** Sort the values and test midpoints between
   consecutive distinct values as thresholds.
3. **How do you prevent overfitting?** Pre-pruning (depth, min samples, min impurity decrease)
   or post-pruning (cost-complexity), validated by cross-validation.
4. **Why don't trees need feature scaling?** Splits depend only on the ordering of values, which
   monotonic rescaling does not change.
5. **Why is the algorithm greedy and is it optimal?** Finding the optimal tree is NP-hard;
   greedy is fast and usually good enough, especially inside ensembles.
