import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_moons, make_regression
from sklearn.ensemble import GradientBoostingClassifier as SkGBC
from sklearn.ensemble import GradientBoostingRegressor as SkGBR

from s3_classical_ml.decision_tree.impl import DecisionTreeRegressor

from .impl import GradientBoostingClassifier, GradientBoostingRegressor


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(n_samples=400, n_features=6, noise=15.0, random_state=0)
    return X[:280], y[:280], X[280:], y[280:]


def test_regressor_matches_sklearn(reg_data):
    Xtr, ytr, Xte, _ = reg_data
    params = dict(n_estimators=30, learning_rate=0.1, max_depth=3)
    ours = GradientBoostingRegressor(**params).fit(Xtr, ytr)
    # criterion="squared_error" makes sklearn choose splits by plain variance reduction, as we do.
    ref = SkGBR(criterion="squared_error", random_state=0, **params).fit(Xtr, ytr)
    np.testing.assert_allclose(ours.predict(Xtr), ref.predict(Xtr), rtol=1e-6)
    np.testing.assert_allclose(ours.predict(Xte), ref.predict(Xte), rtol=1e-6)
    np.testing.assert_allclose(ours.train_loss_, ref.train_score_, rtol=1e-6)


def test_classifier_matches_sklearn():
    X, y = make_moons(400, noise=0.3, random_state=0)
    params = dict(n_estimators=30, learning_rate=0.2, max_depth=2)
    ours = GradientBoostingClassifier(**params).fit(X, y)
    ref = SkGBC(criterion="squared_error", random_state=0, **params).fit(X, y)
    np.testing.assert_allclose(
        ours.decision_function(X), ref.decision_function(X), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-7)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def test_training_loss_decreases_every_round(reg_data):
    Xtr, ytr, _, _ = reg_data
    reg = GradientBoostingRegressor(n_estimators=50).fit(Xtr, ytr)
    assert all(a > b for a, b in zip(reg.train_loss_, reg.train_loss_[1:]))
    X, y = make_moons(300, noise=0.3, random_state=1)
    clf = GradientBoostingClassifier(n_estimators=50).fit(X, y)
    assert all(a > b for a, b in zip(clf.train_loss_, clf.train_loss_[1:]))


def test_first_tree_fits_residuals_of_the_mean(reg_data):
    Xtr, ytr, _, _ = reg_data
    model = GradientBoostingRegressor(n_estimators=1, learning_rate=1.0, max_depth=2).fit(Xtr, ytr)
    by_hand = DecisionTreeRegressor(max_depth=2).fit(Xtr, ytr - ytr.mean())
    np.testing.assert_allclose(model.predict(Xtr), ytr.mean() + by_hand.predict(Xtr))
    assert model.init_ == pytest.approx(ytr.mean())


def test_boosting_beats_baseline_and_a_single_tree(reg_data):
    Xtr, ytr, Xte, yte = reg_data
    boosted = GradientBoostingRegressor(n_estimators=150, max_depth=3).fit(Xtr, ytr)
    mse_boost = np.mean((boosted.predict(Xte) - yte) ** 2)
    mse_tree = np.mean((DecisionTreeRegressor(max_depth=3).fit(Xtr, ytr).predict(Xte) - yte) ** 2)
    mse_mean = np.mean((ytr.mean() - yte) ** 2)
    assert mse_boost < 0.5 * mse_tree < mse_mean


def test_staged_predict_and_shrinkage(reg_data):
    Xtr, ytr, Xte, yte = reg_data
    model = GradientBoostingRegressor(n_estimators=40).fit(Xtr, ytr)
    stages = list(model.staged_predict(Xte))
    assert len(stages) == 40
    np.testing.assert_allclose(stages[-1], model.predict(Xte))
    # A smaller learning rate learns more slowly: after the same few rounds it has fit less.
    slow = GradientBoostingRegressor(n_estimators=5, learning_rate=0.01).fit(Xtr, ytr)
    fast = GradientBoostingRegressor(n_estimators=5, learning_rate=0.5).fit(Xtr, ytr)
    assert slow.train_loss_[-1] > fast.train_loss_[-1]


def test_classifier_generalises_and_validates_input(rng):
    X, y = load_breast_cancer(return_X_y=True)
    idx = rng.permutation(len(y))
    tr, te = idx[:400], idx[400:]
    clf = GradientBoostingClassifier(n_estimators=40, max_depth=2).fit(X[tr], y[tr])
    acc = np.mean(clf.predict(X[te]) == y[te])
    assert acc > 0.93 > max(np.mean(y[te]), 1 - np.mean(y[te]))
    with pytest.raises(ValueError):
        GradientBoostingClassifier().fit(X[:6], np.array([0, 1, 2, 0, 1, 2]))


def test_subsampling_still_learns(reg_data):
    Xtr, ytr, Xte, yte = reg_data
    model = GradientBoostingRegressor(n_estimators=100, subsample=0.5, random_state=0).fit(Xtr, ytr)
    assert np.mean((model.predict(Xte) - yte) ** 2) < 0.4 * np.mean((ytr.mean() - yte) ** 2)
