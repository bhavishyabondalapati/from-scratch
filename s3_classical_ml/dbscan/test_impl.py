import numpy as np
import pytest
from sklearn.cluster import DBSCAN as SkDBSCAN
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.metrics import adjusted_rand_score

from s3_classical_ml.kmeans.impl import KMeans

from .impl import DBSCAN


@pytest.mark.parametrize("eps, min_samples", [(0.1, 3), (0.2, 5), (0.3, 10), (0.15, 1)])
def test_matches_sklearn(eps, min_samples):
    X, _ = make_moons(400, noise=0.08, random_state=0)
    X = np.vstack([X, np.random.default_rng(0).uniform(-1.5, 2.5, size=(40, 2))])  # add noise
    ours = DBSCAN(eps, min_samples).fit(X)
    ref = SkDBSCAN(eps=eps, min_samples=min_samples).fit(X)
    np.testing.assert_array_equal(
        np.sort(ours.core_sample_indices_), np.sort(ref.core_sample_indices_)
    )
    np.testing.assert_array_equal(ours.labels_ == -1, ref.labels_ == -1)  # same noise points
    core = ours.core_sample_indices_
    assert adjusted_rand_score(ref.labels_[core], ours.labels_[core]) == pytest.approx(1.0)


def test_finds_non_convex_shapes_where_kmeans_fails():
    for X, y in (
        make_moons(500, noise=0.05, random_state=1),
        make_circles(500, factor=0.4, noise=0.04, random_state=1),
    ):
        db = DBSCAN(eps=0.15, min_samples=5).fit_predict(X)
        km = KMeans(2, random_state=0).fit_predict(X)
        assert adjusted_rand_score(y, db) > 0.95
        assert adjusted_rand_score(y, km) < 0.6


def test_isolated_points_are_noise_and_cluster_count_is_discovered():
    X, _ = make_blobs(300, centers=[[0, 0], [10, 10], [-10, 10]], cluster_std=0.5, random_state=0)
    outliers = np.array([[30.0, 30.0], [-30.0, -30.0], [5.0, -20.0]])
    labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(np.vstack([X, outliers]))
    assert set(labels[:300]) == {0, 1, 2}  # three clusters, without being told k
    assert (labels[300:] == -1).all()


def test_core_point_definition_by_hand():
    X = np.array([[0.0], [0.5], [1.0], [1.5], [5.0]])
    model = DBSCAN(eps=0.6, min_samples=3).fit(X)
    # 0.0 has neighbours {0, 0.5}: 2 < 3, not core. 0.5 has {0, 0.5, 1.0}: core. etc.
    assert list(model.core_sample_indices_) == [1, 2]
    assert list(model.labels_) == [0, 0, 0, 0, -1]  # 0.0 and 1.5 are border points; 5.0 is noise
