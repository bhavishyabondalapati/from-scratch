import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_moons, make_regression
from sklearn.ensemble import RandomForestClassifier as SkForest

from s3_classical_ml.decision_tree.impl import DecisionTreeClassifier, DecisionTreeRegressor

from .impl import RandomForestClassifier, RandomForestRegressor


@pytest.fixture(scope="module")
def cancer_split():
    X, y = load_breast_cancer(return_X_y=True)
    idx = np.random.default_rng(0).permutation(len(y))
    return X[idx[:400]], y[idx[:400]], X[idx[400:]], y[idx[400:]]


@pytest.fixture(scope="module")
def fitted(cancer_split):
    Xtr, ytr, _, _ = cancer_split
    return RandomForestClassifier(n_estimators=40, random_state=0).fit(Xtr, ytr)


def test_accuracy_is_comparable_to_sklearn(cancer_split, fitted):
    Xtr, ytr, Xte, yte = cancer_split
    ref = SkForest(n_estimators=40, random_state=0).fit(Xtr, ytr)
    acc = np.mean(fitted.predict(Xte) == yte)
    majority = max(np.mean(yte), 1 - np.mean(yte))
    assert acc > 0.93 > majority
    assert abs(acc - ref.score(Xte, yte)) < 0.04


def test_oob_score_estimates_test_accuracy(cancer_split, fitted):
    _, _, Xte, yte = cancer_split
    test_acc = np.mean(fitted.predict(Xte) == yte)
    assert abs(fitted.oob_score_ - test_acc) < 0.05


def test_bootstrap_samples_leave_out_about_37_percent(fitted, cancer_split):
    n = len(cancer_split[0])
    unique_fraction = np.mean([len(np.unique(idx)) / n for idx in fitted.in_bag_])
    assert unique_fraction == pytest.approx(1 - 1 / np.e, abs=0.02)  # 1 - (1 - 1/n)^n -> 0.632


def test_forest_beats_single_tree_on_noisy_data():
    X, y = make_moons(900, noise=0.35, random_state=1)
    Xtr, ytr, Xte, yte = X[:300], y[:300], X[300:], y[300:]
    tree_acc = np.mean(DecisionTreeClassifier(random_state=0).fit(Xtr, ytr).predict(Xte) == yte)
    forest = RandomForestClassifier(n_estimators=60, random_state=0).fit(Xtr, ytr)
    forest_acc = np.mean(forest.predict(Xte) == yte)
    assert forest_acc > tree_acc + 0.02  # averaging de-correlated trees reduces variance


def test_more_trees_do_not_hurt(cancer_split):
    Xtr, ytr, Xte, yte = cancer_split
    few = RandomForestClassifier(n_estimators=1, random_state=3).fit(Xtr, ytr)
    many = RandomForestClassifier(n_estimators=30, random_state=3).fit(Xtr, ytr)
    assert np.mean(many.predict(Xte) == yte) >= np.mean(few.predict(Xte) == yte)


def test_probabilities_and_determinism(cancer_split, fitted):
    Xtr, ytr, Xte, _ = cancer_split
    P = fitted.predict_proba(Xte)
    np.testing.assert_allclose(P.sum(axis=1), 1)
    again = RandomForestClassifier(n_estimators=40, random_state=0).fit(Xtr, ytr)
    np.testing.assert_array_equal(again.predict_proba(Xte), P)  # same seed, same forest
    other = RandomForestClassifier(n_estimators=40, random_state=1).fit(Xtr, ytr)
    assert not np.array_equal(other.predict_proba(Xte), P)


def test_handles_class_missing_from_a_bootstrap_sample():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = np.array([0] * 14 + [1] * 14 + [2] * 2)  # class 2 is rare: some trees never see it
    X[y == 2] += 6
    forest = RandomForestClassifier(n_estimators=80, random_state=0).fit(X, y)
    assert any(len(t.classes_) < 3 for t in forest.trees_)
    assert forest.predict_proba(X).shape == (30, 3)
    assert (forest.predict(X[y == 2]) == 2).all()


def test_regressor_beats_mean_baseline_and_single_tree():
    X, y = make_regression(n_samples=500, n_features=8, n_informative=5, noise=20.0, random_state=0)
    Xtr, ytr, Xte, yte = X[:350], y[:350], X[350:], y[350:]
    forest = RandomForestRegressor(n_estimators=40, random_state=0).fit(Xtr, ytr)
    mse_forest = np.mean((forest.predict(Xte) - yte) ** 2)
    mse_tree = np.mean((DecisionTreeRegressor().fit(Xtr, ytr).predict(Xte) - yte) ** 2)
    mse_mean = np.mean((ytr.mean() - yte) ** 2)
    assert mse_forest < mse_tree < mse_mean
    assert mse_forest < 0.4 * mse_mean
