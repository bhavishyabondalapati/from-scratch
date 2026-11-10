import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, make_blobs, make_circles, make_moons
from sklearn.metrics.pairwise import polynomial_kernel as sk_poly
from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .impl import KernelSVM, LinearSVM, linear_kernel, polynomial_kernel, rbf_kernel


def test_kernels_match_sklearn(rng):
    A, B = rng.normal(size=(8, 3)), rng.normal(size=(5, 3))
    np.testing.assert_allclose(rbf_kernel(A, B, gamma=0.7), sk_rbf(A, B, gamma=0.7))
    np.testing.assert_allclose(
        polynomial_kernel(A, B, degree=3, gamma=0.5, coef0=1.0),
        sk_poly(A, B, degree=3, gamma=0.5, coef0=1.0),
    )
    np.testing.assert_allclose(linear_kernel(A, B), A @ B.T)
    # A valid kernel matrix is symmetric positive semi-definite.
    assert np.linalg.eigvalsh(rbf_kernel(A, A, 0.7)).min() > -1e-10


@pytest.mark.parametrize(
    "kernel, params",
    [
        ("linear", {}),
        ("rbf", {"gamma": 0.5}),
        ("rbf", {"gamma": 2.0}),
        ("poly", {"degree": 3, "gamma": 0.5, "coef0": 1.0}),
    ],
)
@pytest.mark.parametrize("C", [0.1, 1.0, 10.0])
def test_smo_matches_sklearn_svc(kernel, params, C):
    X, y = make_moons(150, noise=0.2, random_state=0)
    ours = KernelSVM(C=C, kernel=kernel, tol=1e-6, **params).fit(X, y)
    ref = SVC(C=C, kernel=kernel, tol=1e-8, **params).fit(X, y)
    grid = np.random.default_rng(1).uniform(-2, 3, size=(200, 2))
    np.testing.assert_allclose(ours.decision_function(grid), ref.decision_function(grid), atol=2e-3)
    np.testing.assert_allclose(ours.decision_function(X), ref.decision_function(X), atol=2e-3)
    assert ours.intercept_ == pytest.approx(ref.intercept_[0], abs=2e-3)
    assert set(ours.support_) == set(ref.support_)


def test_dual_solution_satisfies_kkt_conditions():
    X, y = make_moons(120, noise=0.25, random_state=3)
    C = 1.0
    model = KernelSVM(C=C, kernel="rbf", gamma=1.0, tol=1e-6).fit(X, y)
    y_pm = np.where(y == 1, 1.0, -1.0)
    alpha = model.alpha_
    assert ((alpha >= -1e-12) & (alpha <= C + 1e-12)).all()  # box constraint
    assert alpha @ y_pm == pytest.approx(0, abs=1e-9)  # equality constraint
    margin = y_pm * model.decision_function(X)
    tol = 1e-3
    assert (margin[alpha < 1e-8] >= 1 - tol).all()  # non-support vectors: outside the margin
    assert (margin[alpha > C - 1e-8] <= 1 + tol).all()  # bounded SVs: inside or on the margin
    free = (alpha > 1e-6) & (alpha < C - 1e-6)
    np.testing.assert_allclose(margin[free], 1, atol=tol)  # free SVs: exactly on the margin


def test_kernel_trick_solves_what_a_linear_model_cannot():
    X, y = make_circles(300, factor=0.4, noise=0.08, random_state=0)
    Xtr, ytr, Xte, yte = X[:200], y[:200], X[200:], y[200:]
    linear = KernelSVM(kernel="linear").fit(Xtr, ytr)
    rbf = KernelSVM(kernel="rbf", gamma=1.0).fit(Xtr, ytr)
    assert np.mean(linear.predict(Xte) == yte) < 0.7  # concentric circles: no separating line
    assert np.mean(rbf.predict(Xte) == yte) > 0.97
    assert len(rbf.support_) < len(Xtr)  # the solution is sparse in the training points


def test_hard_margin_on_separable_data_has_few_support_vectors():
    X, y = make_blobs(100, centers=[[-3, -3], [3, 3]], cluster_std=0.8, random_state=0)
    model = KernelSVM(C=1000.0, kernel="linear").fit(X, y)
    assert (model.predict(X) == y).all()
    assert len(model.support_) <= 4
    # Geometric margin = 1 / ||w||; every point is at least that far from the boundary.
    w = model.dual_coef_ @ model.support_vectors_
    distances = np.abs(X @ w + model.intercept_) / np.linalg.norm(w)
    assert distances.min() == pytest.approx(1 / np.linalg.norm(w), rel=1e-3)


def test_linear_svm_subgradient_reaches_the_dual_optimum():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X[:, :8])
    lam = 0.01
    primal = LinearSVM(lam=lam, n_iters=4000).fit(X, y)
    # Same problem for sklearn: C = 1 / (lam * n).
    ref = SVC(kernel="linear", C=1 / (lam * len(y)), tol=1e-8).fit(X, y)
    y_pm = np.where(y == 1, 1.0, -1.0)
    ref_objective = primal.objective(X, y_pm, ref.coef_[0], ref.intercept_[0])
    assert primal.objective_ >= ref_objective - 1e-9  # nothing beats the true optimum...
    assert primal.objective_ <= ref_objective * 1.02  # ...and we get within 2% of it
    assert np.mean(primal.predict(X) == ref.predict(X)) > 0.98
    majority = max(np.mean(y), 1 - np.mean(y))
    assert np.mean(primal.predict(X) == y) > 0.93 > majority


def test_arbitrary_labels_callable_kernel_and_validation():
    X, y = make_blobs(60, centers=[[-2, 0], [2, 0]], cluster_std=0.5, random_state=1)
    labels = np.where(y == 1, "pos", "neg")
    model = KernelSVM(kernel=lambda A, B: A @ B.T).fit(X, labels)
    assert set(model.predict(X)) == {"pos", "neg"} and (model.predict(X) == labels).all()
    with pytest.raises(ValueError):
        KernelSVM().fit(X, np.arange(60) % 3)
    with pytest.raises(ValueError):
        KernelSVM(kernel="quantum").fit(X, y)
