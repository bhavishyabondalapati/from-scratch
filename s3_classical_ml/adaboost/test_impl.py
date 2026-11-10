import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_iris, make_moons
from sklearn.ensemble import AdaBoostClassifier as SkAdaBoost
from sklearn.tree import DecisionTreeClassifier as SkTree

from s3_classical_ml.decision_tree.impl import DecisionTreeClassifier

from .impl import AdaBoostClassifier


def test_matches_sklearn_on_continuous_data():
    X, y = make_moons(300, noise=0.3, random_state=0)
    ours = AdaBoostClassifier(n_estimators=30).fit(X, y)
    ref = SkAdaBoost(SkTree(max_depth=1), n_estimators=30, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.estimator_errors_, ref.estimator_errors_[:30], atol=1e-10)
    np.testing.assert_allclose(ours.estimator_weights_, ref.estimator_weights_[:30], atol=1e-8)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def test_weighted_error_of_each_round_is_below_one_half():
    X, y = make_moons(300, noise=0.3, random_state=1)
    model = AdaBoostClassifier(n_estimators=40).fit(X, y)
    assert all(0 < e < 0.5 for e in model.estimator_errors_)
    assert all(a > 0 for a in model.estimator_weights_)


def test_training_error_respects_the_exponential_bound():
    """Training error <= prod_m 2 * sqrt(err_m * (1 - err_m)) (Freund & Schapire)."""
    X, y = make_moons(300, noise=0.25, random_state=2)
    model = AdaBoostClassifier(n_estimators=60).fit(X, y)
    errors = np.array(model.estimator_errors_)
    bounds = np.cumprod(2 * np.sqrt(errors * (1 - errors)))
    train_errors = [np.mean(pred != y) for pred in model.staged_predict(X)]
    assert all(e <= b + 1e-12 for e, b in zip(train_errors, bounds))
    assert train_errors[-1] < train_errors[0]


def test_boosted_stumps_beat_a_single_stump_and_the_majority_class(rng):
    X, y = load_breast_cancer(return_X_y=True)
    idx = rng.permutation(len(y))
    tr, te = idx[:400], idx[400:]
    stump_acc = np.mean(
        DecisionTreeClassifier(max_depth=1).fit(X[tr], y[tr]).predict(X[te]) == y[te]
    )
    boosted_acc = np.mean(
        AdaBoostClassifier(n_estimators=50).fit(X[tr], y[tr]).predict(X[te]) == y[te]
    )
    majority = max(np.mean(y[te]), 1 - np.mean(y[te]))
    assert boosted_acc > stump_acc > majority
    assert boosted_acc > 0.94


def test_misclassified_samples_gain_weight_by_hand():
    # 1D data where no single threshold is perfect.
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 0, 1])
    model = AdaBoostClassifier(n_estimators=1).fit(X, y)
    # Best stump: threshold 1.5 -> predicts [0, 0, 1, 1, 1]; only x=3 is wrong: err = 1/5.
    assert model.estimator_errors_[0] == pytest.approx(0.2)
    assert model.estimator_weights_[0] == pytest.approx(np.log(0.8 / 0.2))
    # After reweighting, the wrong sample holds exactly half of the total weight.
    w = np.full(5, 0.2) * np.exp(model.estimator_weights_[0] * np.array([0, 0, 0, 1, 0]))
    assert (w / w.sum())[3] == pytest.approx(0.5)


def test_multiclass_and_perfect_learner():
    X, y = load_iris(return_X_y=True)
    model = AdaBoostClassifier(n_estimators=60, max_depth=2).fit(X, y)
    assert np.mean(model.predict(X) == y) > 0.95  # 3 classes; majority baseline is 1/3
    # Separable data: the first stump is perfect, so boosting stops immediately.
    Xs = np.array([[0.0], [1.0], [10.0], [11.0]])
    easy = AdaBoostClassifier(n_estimators=10).fit(Xs, np.array(["lo", "lo", "hi", "hi"]))
    assert len(easy.estimators_) == 1
    assert list(easy.predict([[0.5], [10.5]])) == ["lo", "hi"]
