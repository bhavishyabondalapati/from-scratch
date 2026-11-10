import numpy as np
import pytest
from sklearn.datasets import load_digits, load_iris, load_wine
from sklearn.naive_bayes import GaussianNB as SkGaussianNB
from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB

from .impl import GaussianNB, MultinomialNB


@pytest.mark.parametrize("loader", [load_iris, load_wine])
def test_gaussian_nb_matches_sklearn(loader):
    X, y = loader(return_X_y=True)
    ours, ref = GaussianNB().fit(X, y), SkGaussianNB().fit(X, y)
    np.testing.assert_allclose(ours.theta_, ref.theta_)
    np.testing.assert_allclose(ours.var_, ref.var_)
    np.testing.assert_allclose(ours.class_prior_, ref.class_prior_)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-10)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_multinomial_nb_matches_sklearn(alpha):
    X, y = load_digits(return_X_y=True)  # pixel intensities 0..16 behave like counts
    ours, ref = MultinomialNB(alpha).fit(X, y), SkMultinomialNB(alpha=alpha).fit(X, y)
    np.testing.assert_allclose(ours.feature_log_prob_, ref.feature_log_prob_)
    np.testing.assert_allclose(ours.class_log_prior_, ref.class_log_prior_)
    np.testing.assert_allclose(ours.predict_log_proba(X), ref.predict_log_proba(X), atol=1e-8)
    np.testing.assert_array_equal(ours.predict(X), ref.predict(X))


def test_both_beat_majority_baseline_on_held_out_data(rng):
    X, y = load_digits(return_X_y=True)
    idx = rng.permutation(len(y))
    train, test = idx[:1300], idx[1300:]
    majority = np.bincount(y[test]).max() / len(test)  # ~0.11 for 10 balanced classes
    for model in (GaussianNB(), MultinomialNB()):
        accuracy = np.mean(model.fit(X[train], y[train]).predict(X[test]) == y[test])
        assert accuracy > 0.75 > majority


def test_tiny_spam_example_by_hand():
    # Vocabulary: [free, money, meeting]. Two spam docs, two ham docs.
    X = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 2], [0, 1, 1]])
    y = np.array([1, 1, 0, 0])
    model = MultinomialNB(alpha=1.0).fit(X, y)
    # Spam word totals [3, 3, 0] + 1 smoothing -> [4, 4, 1] / 9
    np.testing.assert_allclose(np.exp(model.feature_log_prob_[1]), [4 / 9, 4 / 9, 1 / 9])
    # Ham totals [0, 1, 3] + 1 -> [1, 2, 4] / 7
    np.testing.assert_allclose(np.exp(model.feature_log_prob_[0]), [1 / 7, 2 / 7, 4 / 7])
    # "free money": P(spam) ∝ 0.5 * (4/9)^2,  P(ham) ∝ 0.5 * (1/7)(2/7)
    spam, ham = 0.5 * (4 / 9) ** 2, 0.5 * (1 / 7) * (2 / 7)
    np.testing.assert_allclose(
        model.predict_proba([[1, 1, 0]])[0], np.array([ham, spam]) / (spam + ham)
    )
    assert list(model.predict([[1, 1, 0], [0, 0, 3]])) == [1, 0]


def test_smoothing_prevents_zero_probabilities():
    X = np.array([[3, 0], [0, 3]])
    y = np.array([0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        unsmoothed = MultinomialNB(alpha=0.0).fit(X, y)
        assert np.isinf(unsmoothed.feature_log_prob_).any()  # log(0)
    smoothed = MultinomialNB(alpha=1.0).fit(X, y)
    assert np.isfinite(smoothed.predict_log_proba([[1, 1]])).all()


def test_long_documents_do_not_underflow(rng):
    X = rng.integers(0, 5, size=(40, 50))
    y = rng.integers(0, 2, size=40)
    model = MultinomialNB().fit(X, y)
    huge_doc = X[:3] * 10_000  # raw probabilities would be ~1e-(millions)
    P = model.predict_proba(huge_doc)
    assert np.isfinite(P).all()
    np.testing.assert_allclose(P.sum(axis=1), 1)


def test_validation_and_constant_feature():
    with pytest.raises(ValueError):
        MultinomialNB().fit(np.array([[1, -1]]), np.array([0]))
    X = np.array([[1.0, 5.0], [2.0, 5.0], [8.0, 5.0], [9.0, 5.0]])  # second feature is constant
    model = GaussianNB().fit(X, np.array([0, 0, 1, 1]))
    assert np.isfinite(model.predict_proba(X)).all()
    assert list(model.predict([[1.5, 5.0], [8.5, 5.0]])) == [0, 1]
