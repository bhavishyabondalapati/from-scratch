import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import cross_val_score as sk_cv
from sklearn.neighbors import KNeighborsClassifier

from .impl import (
    accuracy,
    cross_val_score,
    grid_search,
    kfold,
    stratified_kfold,
    train_val_test_split,
)


def test_split_is_a_disjoint_cover():
    tr, va, te = train_val_test_split(100, val=0.2, test=0.1, seed=0)
    assert (len(tr), len(va), len(te)) == (70, 20, 10)
    assert sorted(np.concatenate([tr, va, te])) == list(range(100))


@pytest.mark.parametrize("n, k", [(10, 2), (23, 5), (100, 10), (7, 7)])
def test_kfold_matches_sklearn_without_shuffle(n, k):
    ours = list(kfold(n, k, shuffle=False))
    ref = list(KFold(k).split(np.zeros(n)))
    for (tr, va), (rtr, rva) in zip(ours, ref):
        np.testing.assert_array_equal(np.sort(tr), rtr)
        np.testing.assert_array_equal(np.sort(va), rva)
    all_val = np.concatenate([va for _, va in ours])
    assert sorted(all_val) == list(range(n))  # every sample validated exactly once
    for tr, va in ours:
        assert not set(tr) & set(va)
    with pytest.raises(ValueError):
        list(kfold(3, 5))


def test_stratified_folds_keep_class_ratios():
    y = np.array([0] * 90 + [1] * 10)  # imbalanced
    for tr, va in stratified_kfold(y, k=5, seed=0):
        assert len(va) == 20 and np.sum(y[va] == 1) == 2
        assert not set(tr) & set(va)
    plain_minority = [np.sum(y[va] == 1) for _, va in kfold(100, 5, seed=3)]
    assert len(set(plain_minority)) > 1  # plain k-fold does NOT guarantee this


def test_cross_val_score_matches_sklearn():
    X, y = load_iris(return_X_y=True)
    folds = list(kfold(len(y), 5, seed=0))
    ours = cross_val_score(lambda: KNeighborsClassifier(5), X, y, accuracy, folds)
    ref = sk_cv(KNeighborsClassifier(5), X, y, cv=[(tr, va) for tr, va in folds])
    np.testing.assert_allclose(ours, ref)


def test_grid_search_matches_sklearn_and_beats_bad_settings():
    X, y = load_breast_cancer(return_X_y=True)
    grid = {"n_neighbors": [1, 3, 7, 15, 31], "weights": ["uniform", "distance"]}
    best, score, results = grid_search(KNeighborsClassifier, grid, X, y, k=5, seed=0)
    folds = list(kfold(len(y), 5, seed=0))
    ref = GridSearchCV(KNeighborsClassifier(), grid, cv=folds).fit(X, y)
    assert score == pytest.approx(ref.best_score_)
    assert best == ref.best_params_
    assert len(results) == 10
    assert score > min(s for _, s in results)
