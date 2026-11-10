import numpy as np
import pytest
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve as sk_calibration_curve

from .impl import (
    average_precision,
    brier_score,
    calibration_curve,
    confusion_matrix,
    expected_calibration_error,
    f1_score,
    precision_recall_f1,
    roc_auc,
    roc_auc_rank,
    roc_curve,
)


def random_binary(rng, n=300, ties=False):
    y = rng.integers(0, 2, size=n)
    scores = y * 0.8 + rng.normal(size=n)
    if ties:
        scores = np.round(scores, 1)  # many tied scores
    return y, scores


def test_confusion_and_prf_match_sklearn(rng):
    for _ in range(20):
        y_true, y_pred = rng.integers(0, 3, size=60), rng.integers(0, 3, size=60)
        np.testing.assert_array_equal(
            confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred)
        )
        for avg in ("macro", "micro", "weighted"):
            assert f1_score(y_true, y_pred, avg) == pytest.approx(
                skm.f1_score(y_true, y_pred, average=avg)
            )
        yb, pb = y_true % 2, y_pred % 2
        p, r, f = precision_recall_f1(yb, pb)
        assert p == pytest.approx(skm.precision_score(yb, pb))
        assert r == pytest.approx(skm.recall_score(yb, pb))
        assert f == pytest.approx(skm.f1_score(yb, pb)) == pytest.approx(f1_score(yb, pb))


def test_degenerate_cases():
    assert precision_recall_f1([0, 0, 1], [0, 0, 0]) == (
        0.0,
        0.0,
        0.0,
    )  # nothing predicted positive
    assert confusion_matrix(["a", "b"], ["b", "b"]).tolist() == [[0, 1], [0, 1]]
    with pytest.raises(ValueError):
        f1_score([0, 1], [0, 1], average="vibes")


@pytest.mark.parametrize("ties", [False, True])
def test_roc_and_pr_match_sklearn(rng, ties):
    for _ in range(20):
        y, s = random_binary(rng, ties=ties)
        assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s))
        assert roc_auc_rank(y, s) == pytest.approx(skm.roc_auc_score(y, s))
        assert average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s))
        fpr, tpr, _ = roc_curve(y, s)
        rfpr, rtpr, _ = skm.roc_curve(y, s, drop_intermediate=False)
        np.testing.assert_allclose(fpr, rfpr)
        np.testing.assert_allclose(tpr, rtpr)


def test_auc_reference_points(rng):
    y = rng.integers(0, 2, size=1000)
    assert roc_auc(y, y + 0.0) == 1.0  # perfect ranking
    assert roc_auc(y, -y + 0.0) == 0.0  # perfectly wrong
    assert roc_auc(y, rng.random(1000)) == pytest.approx(0.5, abs=0.05)  # random scores
    assert average_precision(y, rng.random(1000)) == pytest.approx(
        y.mean(), abs=0.05
    )  # PR baseline = prevalence


def test_calibration_matches_sklearn_and_ece(rng):
    p = rng.random(5000)
    y = (rng.random(5000) < p).astype(int)  # perfectly calibrated by construction
    frac, mean_pred, _ = calibration_curve(y, p, 10)
    rfrac, rmean = sk_calibration_curve(y, p, n_bins=10)
    np.testing.assert_allclose(frac, rfrac)
    np.testing.assert_allclose(mean_pred, rmean)
    assert expected_calibration_error(y, p) < 0.03
    overconfident = np.clip(0.5 + 2 * (p - 0.5), 0, 1)
    assert expected_calibration_error(y, overconfident) > 0.1
    assert brier_score(y, p) == pytest.approx(skm.brier_score_loss(y, p))
    assert brier_score(y, p) < brier_score(y, overconfident)
