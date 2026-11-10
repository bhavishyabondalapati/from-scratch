import numpy as np
import pytest
from scipy.stats import entropy as scipy_entropy
from sklearn.datasets import load_breast_cancer, load_iris, make_regression
from sklearn.tree import DecisionTreeClassifier as SkTreeClassifier
from sklearn.tree import DecisionTreeRegressor as SkTreeRegressor

from .impl import DecisionTreeClassifier, DecisionTreeRegressor, entropy, gini, variance_from_stats


def test_impurity_measures(rng):
    assert gini(np.array([10.0, 0.0])) == 0 and entropy(np.array([10.0, 0.0])) == 0
    assert gini(np.array([5.0, 5.0])) == pytest.approx(0.5)
    assert entropy(np.array([5.0, 5.0])) == pytest.approx(1.0)
    assert entropy(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(2.0)
    for _ in range(20):
        counts = rng.integers(0, 20, size=4).astype(float) + 1
        assert entropy(counts) == pytest.approx(scipy_entropy(counts, base=2))
        p = counts / counts.sum()
        assert gini(counts) == pytest.approx(1 - np.sum(p**2))
    y, w = rng.normal(size=30), rng.random(30)
    stats = np.array([w.sum(), (w * y).sum(), (w * y**2).sum()])
    assert variance_from_stats(stats) == pytest.approx(
        np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    )


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_root_split_is_as_good_as_sklearn(criterion):
    """A depth-1 tree makes one greedy split; ours must achieve the same impurity."""
    for loader in (load_iris, load_breast_cancer):
        X, y = loader(return_X_y=True)
        ours = DecisionTreeClassifier(criterion=criterion, max_depth=1).fit(X, y)
        ref = SkTreeClassifier(criterion=criterion, max_depth=1, random_state=0).fit(X, y).tree_
        ref_children = (
            ref.weighted_n_node_samples[1] * ref.impurity[1]
            + ref.weighted_n_node_samples[2] * ref.impurity[2]
        ) / len(y)
        root = ours.root_
        our_children = (
            root.left.n_samples * root.left.impurity + root.right.n_samples * root.right.impurity
        ) / len(y)
        assert root.impurity == pytest.approx(ref.impurity[0])
        assert our_children == pytest.approx(ref_children)


def test_regressor_matches_sklearn_exactly_on_continuous_data():
    X, y = make_regression(n_samples=300, n_features=5, noise=10.0, random_state=0)
    for depth in (1, 2, 4, None):
        ours = DecisionTreeRegressor(max_depth=depth).fit(X, y)
        ref = SkTreeRegressor(max_depth=depth, random_state=0).fit(X, y)
        np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-9)
        assert ours.n_leaves_ == ref.get_n_leaves()
    Xq = np.random.default_rng(1).normal(size=(100, 5))
    np.testing.assert_allclose(
        DecisionTreeRegressor(max_depth=4).fit(X, y).predict(Xq),
        SkTreeRegressor(max_depth=4, random_state=0).fit(X, y).predict(Xq),
    )


def test_classifier_generalises_like_sklearn(rng):
    X, y = load_breast_cancer(return_X_y=True)
    idx = rng.permutation(len(y))
    train, test = idx[:400], idx[400:]
    for criterion in ("gini", "entropy"):
        ours = DecisionTreeClassifier(criterion=criterion, max_depth=4).fit(X[train], y[train])
        ref = SkTreeClassifier(criterion=criterion, max_depth=4, random_state=0).fit(
            X[train], y[train]
        )
        acc = np.mean(ours.predict(X[test]) == y[test])
        assert acc > 0.88 > max(np.mean(y[test]), 1 - np.mean(y[test]))  # beats majority class
        assert abs(acc - ref.score(X[test], y[test])) < 0.05
        assert np.mean(ours.predict(X[train]) == y[train]) == pytest.approx(
            ref.score(X[train], y[train]), abs=0.02
        )


def test_unlimited_depth_memorises_and_hyperparameters_limit_growth():
    X, y = load_iris(return_X_y=True)
    X, keep = np.unique(X, axis=0, return_index=True)  # identical rows cannot be separated
    y = y[keep]
    full = DecisionTreeClassifier().fit(X, y)
    assert (full.predict(X) == y).all()
    shallow = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert shallow.depth() == 2 and shallow.n_leaves_ <= 4
    big_leaves = DecisionTreeClassifier(min_samples_leaf=20).fit(X, y)

    def leaf_sizes(node):
        return [node.n_samples] if node.is_leaf else leaf_sizes(node.left) + leaf_sizes(node.right)

    assert min(leaf_sizes(big_leaves.root_)) >= 20
    assert sum(leaf_sizes(big_leaves.root_)) == len(X)


def test_predict_proba_and_string_labels():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0]])
    y = np.array(["a", "a", "a", "b", "b", "b"])
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert stump.root_.threshold == pytest.approx(2.5)  # midpoint between 2 and 3
    assert list(stump.predict([[-5.0], [99.0]])) == ["a", "b"]
    forced = DecisionTreeClassifier(max_depth=1, min_samples_leaf=2).fit(X[:5], y[:5])
    np.testing.assert_allclose(forced.predict_proba(X[:5]).sum(axis=1), 1)


def test_sample_weights_change_the_split():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    # Heavily weighted points 1 and 2 dominate: the best single cut separates them.
    w = np.array([0.01, 10.0, 10.0, 0.01])
    stump = DecisionTreeClassifier(max_depth=1).fit(X, y, sample_weight=w)
    assert stump.root_.threshold == pytest.approx(1.5)
    # Weighting is equivalent to repeating samples.
    reps = np.array([1, 3, 2, 5])
    weighted = DecisionTreeRegressor(max_depth=1).fit(X, y.astype(float) * 7, sample_weight=reps)
    repeated = DecisionTreeRegressor(max_depth=1).fit(
        np.repeat(X, reps, axis=0), np.repeat(y * 7.0, reps)
    )
    np.testing.assert_allclose(weighted.predict(X), repeated.predict(X))


def test_xor_needs_depth_two():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float).repeat(5, axis=0)
    X += np.random.default_rng(0).normal(scale=0.01, size=X.shape)
    y = (X[:, 0].round() != X[:, 1].round()).astype(int)
    assert np.mean(DecisionTreeClassifier(max_depth=1).fit(X, y).predict(X) == y) < 0.8
    assert (DecisionTreeClassifier().fit(X, y).predict(X) == y).all()


def test_invalid_criterion():
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="vibes")
