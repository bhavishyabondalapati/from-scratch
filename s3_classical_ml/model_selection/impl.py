"""Model selection: train/val/test split, k-fold (and stratified) CV, grid search."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np


def train_val_test_split(
    n: int, val: float = 0.15, test: float = 0.15, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle indices 0..n-1 and cut them into three disjoint parts.

    train: fit parameters. val: choose hyperparameters / early stopping.
    test: touched ONCE at the very end for an unbiased estimate.
    """
    idx = np.random.default_rng(seed).permutation(n)
    n_test, n_val = int(round(test * n)), int(round(val * n))
    return idx[n_test + n_val :], idx[n_test : n_test + n_val], idx[:n_test]


def kfold(
    n: int, k: int = 5, shuffle: bool = True, seed: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) k times; every sample is in exactly one val fold.

    Fold sizes differ by at most one (the first n % k folds get one extra).
    """
    if not 2 <= k <= n:
        raise ValueError("need 2 <= k <= n")
    idx = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    sizes = np.full(k, n // k)
    sizes[: n % k] += 1
    start = 0
    for size in sizes:
        val = idx[start : start + size]
        yield np.concatenate([idx[:start], idx[start + size :]]), val
        start += size


def stratified_kfold(
    y: np.ndarray, k: int = 5, seed: int | None = None
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Like kfold, but every fold keeps the class proportions of y.

    Deal each class's (shuffled) indices round-robin into the k folds.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    offset = 0
    for cls in np.unique(y):
        members = rng.permutation(np.flatnonzero(y == cls))
        for i, m in enumerate(members):
            folds[(offset + i) % k].append(int(m))
        offset += len(members)  # continue the round-robin so fold sizes stay balanced
    all_idx = np.arange(len(y))
    for fold in folds:
        val = np.array(sorted(fold))
        yield np.setdiff1d(all_idx, val), val


def cross_val_score(
    make_model: Callable[[], Any],
    X: np.ndarray,
    y: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], float],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Fit a FRESH model on each training fold and score it on the matching val fold."""
    scores = []
    for tr, va in folds:
        model = make_model().fit(X[tr], y[tr])
        scores.append(scorer(y[va], model.predict(X[va])))
    return np.array(scores)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def grid_search(
    make_model: Callable[..., Any],
    param_grid: dict[str, list[Any]],
    X: np.ndarray,
    y: np.ndarray,
    scorer: Callable[[np.ndarray, np.ndarray], float] = accuracy,
    k: int = 5,
    seed: int = 0,
) -> tuple[dict[str, Any], float, list[tuple[dict[str, Any], float]]]:
    """Try every combination of hyperparameters with k-fold CV; higher score wins.

    All combinations use the SAME folds so their scores are comparable.
    Returns (best params, best mean score, all results).
    """
    folds = list(kfold(len(X), k, seed=seed))
    names = list(param_grid)
    results = []
    for values in itertools.product(*(param_grid[name] for name in names)):
        params = dict(zip(names, values))
        score = float(cross_val_score(lambda p=params: make_model(**p), X, y, scorer, folds).mean())
        results.append((params, score))
    best_params, best_score = max(results, key=lambda r: r[1])  # first one wins ties
    return best_params, best_score, results
