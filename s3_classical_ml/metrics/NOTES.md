# Classification metrics

## Intuition
Accuracy alone lies when classes are imbalanced: on a dataset with 1% fraud, "never fraud" is
99% accurate and useless. Other metrics ask sharper questions:

- **Confusion matrix:** the full table of (true class, predicted class) counts.
- **Precision:** of the alarms we raised, how many were real? (cost of false alarms)
- **Recall:** of the real cases, how many did we catch? (cost of misses)
- **F1:** harmonic mean; high only if both are high.
- **ROC curve / AUC:** sweep the threshold; plot true-positive rate vs false-positive rate.
  AUC = probability a random positive is scored above a random negative. Threshold-free.
- **PR curve / average precision:** precision vs recall over thresholds; much more informative
  than ROC when positives are rare.
- **Calibration:** when the model says 70%, does it happen 70% of the time?

## Core math
- `P = TP/(TP+FP)`, `R = TP/(TP+FN)`, `F1 = 2PR/(P+R)`, `FPR = FP/(FP+TN)`.
- Macro F1 = mean over classes; micro F1 = pooled counts (= accuracy for single-label);
  weighted = support-weighted mean.
- `AUC = (Σ ranks of positives − n₊(n₊+1)/2) / (n₊n₋)` (Mann-Whitney U), ties count half.
- `AP = Σₖ (Rₖ − Rₖ₋₁) Pₖ`; a random classifier's AP ≈ prevalence (ROC-AUC = 0.5).
- ECE `= Σ_b (n_b/n) |acc_b − conf_b|`; Brier `= mean((p − y)²)`.

## Complexity
O(n) for counts; O(n log n) for curves/AUC (sorting).

## Common pitfalls
- Accuracy on imbalanced data; ROC-AUC looking great while precision at useful thresholds is poor.
- Forgetting that ties in scores must move together along the curve.
- Comparing AP across datasets with different prevalence.
- Picking the threshold on the test set.
- Macro vs micro averaging give very different stories with rare classes.
- Treating high AUC as good calibration (AUC only measures ranking).

## When to use vs. alternatives
- Balanced classes, equal error costs → accuracy is fine.
- Costs of FP vs FN differ → precision/recall at a chosen threshold, or expected cost.
- Ranking quality → ROC-AUC (balanced) or PR-AUC (rare positives).
- Probabilities used for decisions → calibration curve, ECE, Brier, log-loss.

## Interview questions
1. **Precision vs recall — when do you favour each?** Recall for screening (missing cancer is
   costly); precision when false alarms are costly (spam filtering of important email).
2. **What does ROC-AUC mean probabilistically?** P(score of a random positive > score of a random
   negative).
3. **Why PR-AUC for imbalanced data?** ROC's FPR has a huge TN denominator and hides many false
   positives; precision exposes them.
4. **Macro vs micro F1?** Macro treats classes equally (sensitive to rare classes); micro treats
   samples equally.
5. **How do you check and fix calibration?** Reliability diagram / ECE; fix with Platt scaling or
   isotonic regression on a validation set.
