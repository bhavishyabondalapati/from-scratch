import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.datasets import load_iris, make_blobs
from sklearn.metrics import adjusted_rand_score

from .impl import KMeans, kmeans_plus_plus


def test_recovers_well_separated_blobs():
    X, y = make_blobs(600, centers=4, cluster_std=0.6, random_state=0)
    model = KMeans(4, random_state=0).fit(X)
    assert adjusted_rand_score(y, model.labels_) == pytest.approx(1.0)


def test_inertia_matches_sklearn_on_iris():
    X = load_iris().data
    for k in (2, 3, 5):
        ours = KMeans(k, n_init=10, random_state=0).fit(X)
        ref = SkKMeans(k, n_init=10, random_state=0).fit(X)
        assert ours.inertia_ == pytest.approx(ref.inertia_, rel=1e-3)
        # Same local optimum => same partition (labels may be permuted).
        assert adjusted_rand_score(ref.labels_, ours.labels_) > 0.99


def test_lloyd_never_increases_inertia(rng):
    X = rng.normal(size=(300, 2))
    model = KMeans(6, n_init=1, init="random", random_state=1).fit(X)
    h = model.inertia_history_
    assert all(a >= b - 1e-9 for a, b in zip(h, h[1:]))


def test_fixed_point_properties(rng):
    X = rng.normal(size=(200, 3))
    model = KMeans(5, random_state=0).fit(X)
    for c in range(5):  # every centre is the mean of its members
        np.testing.assert_allclose(model.cluster_centers_[c], X[model.labels_ == c].mean(axis=0))
    np.testing.assert_array_equal(
        model.predict(X), model.labels_
    )  # every point is nearest its own centre
    expected = sum(
        ((X[model.labels_ == c] - model.cluster_centers_[c]) ** 2).sum() for c in range(5)
    )
    assert model.inertia_ == pytest.approx(expected)


def test_kmeans_plus_plus_spreads_centres_out():
    X, _ = make_blobs(1000, centers=10, cluster_std=0.3, center_box=(-50, 50), random_state=3)
    rng = np.random.default_rng(0)

    def blobs_hit(init_centres):
        return len({tuple(np.round(c / 10)) for c in init_centres})

    hits_pp = np.mean([blobs_hit(kmeans_plus_plus(X, 10, rng)) for _ in range(30)])
    hits_rand = np.mean([blobs_hit(X[rng.choice(len(X), 10, replace=False)]) for _ in range(30)])
    assert hits_pp > hits_rand  # ++ seeding lands in more distinct blobs
    one_shot = [
        KMeans(10, n_init=1, init=i, random_state=s).fit(X).inertia_
        for i in ("k-means++", "random")
        for s in range(10)
    ]
    assert np.mean(one_shot[:10]) < np.mean(one_shot[10:])  # and gives better single runs


def test_more_clusters_never_worse_and_beats_trivial_baseline(rng):
    X = load_iris().data
    inertias = [KMeans(k, random_state=0).fit(X).inertia_ for k in range(1, 7)]
    assert all(a >= b for a, b in zip(inertias, inertias[1:]))
    assert inertias[0] == pytest.approx(((X - X.mean(0)) ** 2).sum())  # k=1: total variance
    assert inertias[2] < 0.2 * inertias[0]
    with pytest.raises(ValueError):
        KMeans(init="nope")
