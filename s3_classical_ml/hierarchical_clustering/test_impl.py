import numpy as np
import pytest
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage as scipy_linkage
from sklearn.datasets import load_iris, make_blobs, make_moons
from sklearn.metrics import adjusted_rand_score

from .impl import AgglomerativeClustering, cut_tree, linkage_matrix

METHODS = ["single", "complete", "average", "ward"]


@pytest.mark.parametrize("method", METHODS)
def test_merge_heights_match_scipy(rng, method):
    X = rng.normal(size=(40, 3))  # continuous data: no tied distances
    ours = linkage_matrix(X, method)
    ref = scipy_linkage(X, method)
    np.testing.assert_allclose(ours[:, 2], ref[:, 2], rtol=1e-9)  # merge distances
    np.testing.assert_array_equal(ours[:, 3], ref[:, 3])  # cluster sizes
    np.testing.assert_array_equal(ours[:, :2], ref[:, :2])  # which clusters merged


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("k", [2, 3, 5])
def test_flat_clusters_match_scipy(method, k):
    X = load_iris().data + np.random.default_rng(0).normal(scale=1e-6, size=(150, 4))
    ours = cut_tree(linkage_matrix(X, method), k)
    ref = fcluster(scipy_linkage(X, method), k, criterion="maxclust")
    assert len(np.unique(ours)) == k
    assert adjusted_rand_score(ref, ours) == pytest.approx(1.0)


def test_heights_are_monotone_for_these_linkages(rng):
    X = rng.normal(size=(60, 2))
    for method in METHODS:
        h = linkage_matrix(X, method)[:, 2]
        assert all(a <= b + 1e-12 for a, b in zip(h, h[1:]))


def test_single_linkage_follows_chains_and_ward_finds_blobs():
    X, y = make_moons(200, noise=0.04, random_state=0)
    assert adjusted_rand_score(
        y, AgglomerativeClustering(2, "single").fit_predict(X)
    ) == pytest.approx(1.0)
    Xb, yb = make_blobs(300, centers=4, cluster_std=0.7, random_state=1)
    assert adjusted_rand_score(yb, AgglomerativeClustering(4, "ward").fit_predict(Xb)) > 0.95


def test_invalid_linkage():
    with pytest.raises(ValueError):
        linkage_matrix(np.zeros((3, 2)), "median-ish")
