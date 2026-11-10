"""Classification metrics: confusion matrix, precision/recall/F1, ROC-AUC, PR-AUC, calibration."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None
) -> np.ndarray:
    """C[i, j] = number of samples whose TRUE class is labels[i] and PREDICTED class is labels[j]."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    index = {lab: i for i, lab in enumerate(labels.tolist())}
    C = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        C[index[t], index[p]] += 1
    return C


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, positive=1
) -> tuple[float, float, float]:
    """For one positive class:
    precision = TP / (TP + FP): of everything flagged, how much was right?
    recall    = TP / (TP + FN): of everything real, how much did we find?
    F1        = harmonic mean of the two (low if either is low).
    Undefined ratios (0/0) are reported as 0.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> float:
    """average='binary' (positive class 1), 'macro' (unweighted mean over classes),
    'micro' (pool all TP/FP/FN; equals accuracy for single-label problems),
    'weighted' (mean weighted by class frequency)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if average == "binary":
        return precision_recall_f1(y_true, y_pred, 1)[2]
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if average == "micro":
        return float(np.mean(y_true == y_pred))
    f1s = np.array([precision_recall_f1(y_true, y_pred, c)[2] for c in labels])
    if average == "macro":
        return float(f1s.mean())
    if average == "weighted":
        support = np.array([np.sum(y_true == c) for c in labels])
        return float((f1s * support).sum() / support.sum())
    raise ValueError(f"unknown average {average!r}")


def _sorted_counts(y_true: np.ndarray, scores: np.ndarray):
    """Cumulative TP and FP when we lower the threshold past each DISTINCT score."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], (y_true[order] == 1).astype(float)
    last_of_group = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # ties move together
    tps = np.cumsum(y)[last_of_group]
    fps = (last_of_group + 1) - tps
    return tps, fps, s[last_of_group]


def roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(false positive rate, true positive rate, thresholds) as the threshold sweeps down."""
    tps, fps, thresholds = _sorted_counts(y_true, scores)
    tpr = np.r_[0.0, tps / tps[-1]]
    fpr = np.r_[0.0, fps / fps[-1]]
    return fpr, tpr, np.r_[np.inf, thresholds]


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (trapezoid rule).

    Equals the probability that a random positive gets a higher score than a
    random negative (ties count half).
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.trapezoid(tpr, fpr))


def roc_auc_rank(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Same number via the Mann-Whitney U statistic: (sum of positive ranks - n+(n+ + 1)/2) / (n+ n-)."""
    y_true, scores = np.asarray(y_true) == 1, np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):  # average ranks over ties
        tie = scores == v
        ranks[tie] = ranks[tie].mean()
    n_pos, n_neg = y_true.sum(), (~y_true).sum()
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_recall_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tps, fps, _ = _sorted_counts(y_true, scores)
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return np.r_[1.0, precision], np.r_[0.0, recall]


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC as average precision: sum over thresholds of (R_k - R_{k-1}) * P_k.
    (Step-wise, no interpolation; the standard definition.)"""
    precision, recall = precision_recall_curve(y_true, scores)
    return float(np.sum(np.diff(recall) * precision[1:]))


def calibration_curve(
    y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin predicted probabilities into equal-width bins; for each non-empty bin
    return (fraction of positives, mean predicted probability, count).
    A calibrated model has fraction ≈ mean prediction in every bin."""
    y_true, prob = np.asarray(y_true, dtype=float), np.asarray(prob, dtype=float)
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    frac, mean_pred, count = [], [], []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            frac.append(y_true[m].mean())
            mean_pred.append(prob[m].mean())
            count.append(m.sum())
    return np.array(frac), np.array(mean_pred), np.array(count)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Weighted average gap |accuracy - confidence| across bins."""
    frac, mean_pred, count = calibration_curve(y_true, prob, n_bins)
    return float(np.sum(count * np.abs(frac - mean_pred)) / count.sum())


def brier_score(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Mean squared error of probabilities: rewards both calibration and sharpness."""
    return float(np.mean((np.asarray(prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))
