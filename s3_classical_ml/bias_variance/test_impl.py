import numpy as np
import pytest

from .impl import bias_variance_decomposition, polynomial_features, regularization_path, save_plots


def test_polynomial_features():
    np.testing.assert_array_equal(
        polynomial_features(np.array([2.0, 3.0]), 3), [[2, 4, 8], [3, 9, 27]]
    )


def test_decomposition_adds_up():
    for degree in (1, 3, 9):
        p = bias_variance_decomposition(degree, n_datasets=300)
        total = p["bias_sq"] + p["variance"] + p["noise"]
        assert p["test_mse"] == pytest.approx(total, rel=0.1)


def test_bias_falls_and_variance_rises_with_complexity():
    simple = bias_variance_decomposition(1)
    medium = bias_variance_decomposition(4)
    complex_ = bias_variance_decomposition(12)
    assert simple["bias_sq"] > medium["bias_sq"]  # a line cannot follow a sine wave
    assert complex_["variance"] > medium["variance"] > simple["variance"]
    # The sweet spot in the middle beats both extremes.
    assert medium["test_mse"] < simple["test_mse"]
    assert medium["test_mse"] < complex_["test_mse"]


def test_regularization_curve_is_u_shaped():
    alphas, train, val, norms = regularization_path()
    assert np.all(np.diff(train) >= -1e-9)  # more penalty, worse fit to training data
    assert np.all(np.diff(norms) <= 1e-9)  # and smaller weights
    best = int(np.argmin(val))
    assert 0 < best < len(alphas) - 1  # the optimum is strictly inside the range
    assert val[best] < 0.7 * val[0] and val[best] < 0.7 * val[-1]


def test_plots_are_written(tmp_path):
    save_plots(tmp_path)
    assert (tmp_path / "bias_variance.png").stat().st_size > 1000
    assert (tmp_path / "regularization.png").stat().st_size > 1000
