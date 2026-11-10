# AdaBoost

## Intuition
Train a sequence of very simple classifiers ("weak learners", usually one-question decision
stumps). After each one, **turn up the importance of the examples it got wrong**, so the next one
is forced to focus on them. At the end, let all of them vote, with more accurate ones getting a
louder voice.

A study-group analogy: after each practice exam you spend the next session on the questions you
missed. No single session covers everything, but together they do.

## Core math
With sample weights `wᵢ` (sum to 1) and learner `hₘ`:
- Weighted error: `εₘ = Σ wᵢ · 1[hₘ(xᵢ) ≠ yᵢ]`.
- Learner weight (SAMME, K classes): `αₘ = η [ln((1 − εₘ)/εₘ) + ln(K − 1)]`.
  (Classic binary AdaBoost uses `½ ln((1 − ε)/ε)`; same model, different scaling.)
- Reweight: `wᵢ ← wᵢ · exp(αₘ · 1[wrong])`, then normalise. After the update, the samples
  the learner got wrong hold exactly half of the total weight (for K = 2).
- Prediction: `argmax_k Σₘ αₘ · 1[hₘ(x) = k]`.
- AdaBoost is forward stagewise additive modelling under the **exponential loss**
  `L = exp(−y F(x))`.
- Training error bound: `≤ Πₘ 2√(εₘ(1 − εₘ))`, which shrinks exponentially as long as every
  learner is a bit better than chance.

## Complexity
O(M · cost of fitting one weak learner). With stumps: O(M · d · n log n). Predict: O(M).

## Common pitfalls
- **Sensitive to label noise and outliers:** mislabelled points keep gaining weight
  exponentially until the model contorts to fit them.
- Weak learner must beat random guessing (ε < 1 − 1/K) or the round is useless/harmful.
- Weak learner too strong (deep trees) → ε = 0 on the first round and boosting does nothing.
- The weak learner must actually *use* the sample weights (or be trained on a weighted resample).
- Output scores are not calibrated probabilities.

## When to use vs. alternatives
- Historically important, simple, few hyperparameters, good on clean tabular data and for
  face detection (Viola-Jones).
- Gradient boosting generalises the idea to any differentiable loss and is more robust; it is
  the usual choice today.
- Noisy labels → random forest or gradient boosting with a robust loss.

## Interview questions
1. **How does AdaBoost differ from gradient boosting?** AdaBoost reweights samples and is tied
   to exponential loss; gradient boosting fits new learners to loss gradients for any loss.
2. **What is a weak learner?** A model only slightly better than chance, e.g. a decision stump.
3. **Why is AdaBoost sensitive to outliers?** Exponential loss puts enormous weight on badly
   misclassified points.
4. **How are the learner weights chosen?** `α ∝ ln((1 − ε)/ε)`: zero at chance level, growing
   as the weighted error falls.
5. **Does AdaBoost overfit?** Test error often keeps improving after training error hits zero
   (margins grow), but with noisy data it can overfit badly.
