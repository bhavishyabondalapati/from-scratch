import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sklearn.datasets import load_iris, make_moons
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .impl import KNNClassifier, KNNRegressor, minkowski_distances, pairwise_sq_distances


def test_distance_tables_match_scipy(rng):
    A, B = rng.normal(size=(15, 4)), rng.normal(size=(9, 4))
    np.testing.assert_allclose(pairwise_sq_distances(A, B), cdist(A, B, "sqeuclidean"), atol=1e-10)
    for p in (1, 2, 3):
        np.testing.assert_allclose(minkowski_distances(A, B, p), cdist(A, B, "minkowski", p=p))
    assert (pairwise_sq_distances(A, A) >= 0).all()


@pytest.mark.parametrize("k", [1, 3, 7, 15])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classifier_matches_sklearn(rng, k, weights):
    X = rng.normal(size=(200, 3))  # continuous data: no distance ties
    y = (X[:, 0] + X[:, 1] ** 2 + 0.5 * rng.normal(size=200) > 0.5).astype(int) + (X[:, 2] > 1)
    Xq = rng.normal(size=(80, 3))
    ours = KNNClassifier(k, weights).fit(X, y)
    ref = KNeighborsClassifier(k, weights=weights, algorithm="brute").fit(X, y)
    np.testing.assert_array_equal(ours.predict(Xq), ref.predict(Xq))
    np.testing.assert_allclose(ours.predict_proba(Xq), ref.predict_proba(Xq))
    d_ours, i_ours = ours.kneighbors(Xq)
    d_ref, i_ref = ref.kneighbors(Xq)
    np.testing.assert_array_equal(i_ours, i_ref)
    np.testing.assert_allclose(d_ours, d_ref, atol=1e-10)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_regressor_matches_sklearn(rng, weights):
    X = rng.uniform(-3, 3, size=(150, 2))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.1 * rng.normal(size=150)
    Xq = rng.uniform(-3, 3, size=(60, 2))
    ours = KNNRegressor(5, weights).fit(X, y).predict(Xq)
    ref = KNeighborsRegressor(5, weights=weights).fit(X, y).predict(Xq)
    np.testing.assert_allclose(ours, ref)


def test_k1_memorises_training_set_and_k_n_predicts_majority():
    X, y = load_iris(return_X_y=True)
    X, idx = np.unique(X, axis=0, return_index=True)  # drop duplicate rows
    y = y[idx]
    assert (KNNClassifier(1).fit(X, y).predict(X) == y).all()
    everyone = KNNClassifier(len(X)).fit(X, y).predict(X[:5])
    assert (everyone == np.bincount(y).argmax()).all()
    # Distance weighting returns the exact label when the query IS a training point.
    assert (KNNClassifier(10, "distance").fit(X, y).predict(X) == y).all()


def test_beats_baseline_on_nonlinear_data():
    X, y = make_moons(600, noise=0.25, random_state=0)
    train, test = slice(0, 400), slice(400, None)
    accuracy = np.mean(KNNClassifier(15).fit(X[train], y[train]).predict(X[test]) == y[test])
    assert accuracy > 0.9  # a linear model gets ~0.85; majority class gets ~0.5


def test_string_labels_and_validation():
    X = np.array([[0.0], [0.1], [5.0], [5.1]])
    y = np.array(["cat", "cat", "dog", "dog"])
    assert list(KNNClassifier(3).fit(X, y).predict([[0.05], [4.9]])) == ["cat", "dog"]
    with pytest.raises(ValueError):
        KNNClassifier(3, weights="nope")
