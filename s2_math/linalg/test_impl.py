import math

import numpy as np
import pytest

from .impl import (
    back_substitution,
    cosine_similarity,
    determinant,
    dot,
    identity,
    inverse,
    least_squares,
    matmul,
    norm,
    power_iteration,
    project,
    qr_gram_schmidt,
    rank,
    solve,
    transpose,
)


def test_vector_ops_match_numpy(rng):
    for _ in range(50):
        n = rng.integers(1, 8)
        u, v = rng.normal(size=n), rng.normal(size=n)
        assert dot(u.tolist(), v.tolist()) == pytest.approx(u @ v)
        for p in (1, 2, 3, math.inf):
            assert norm(u.tolist(), p) == pytest.approx(np.linalg.norm(u, p))
        expected_cos = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        assert cosine_similarity(u.tolist(), v.tolist()) == pytest.approx(expected_cos)
        # The residual u - proj is perpendicular to v.
        proj = np.array(project(u.tolist(), v.tolist()))
        assert (u - proj) @ v == pytest.approx(0, abs=1e-9)
    with pytest.raises(ValueError):
        dot([1, 2], [1])


def test_matmul_and_transpose_match_numpy(rng):
    for _ in range(50):
        n, m, p = rng.integers(1, 6, size=3)
        A, B = rng.normal(size=(n, m)), rng.normal(size=(m, p))
        np.testing.assert_allclose(matmul(A.tolist(), B.tolist()), A @ B)
        np.testing.assert_allclose(transpose(A.tolist()), A.T)
    with pytest.raises(ValueError):
        matmul([[1, 2]], [[1, 2]])


def test_solve_inverse_determinant_match_numpy(rng):
    for _ in range(100):
        n = rng.integers(1, 7)
        A, b = rng.normal(size=(n, n)), rng.normal(size=n)
        np.testing.assert_allclose(solve(A.tolist(), b.tolist()), np.linalg.solve(A, b), atol=1e-8)
        np.testing.assert_allclose(inverse(A.tolist()), np.linalg.inv(A), atol=1e-8)
        assert determinant(A.tolist()) == pytest.approx(np.linalg.det(A), rel=1e-8, abs=1e-10)
        np.testing.assert_allclose(matmul(A.tolist(), inverse(A.tolist())), identity(n), atol=1e-8)


def test_pivoting_handles_zero_on_the_diagonal():
    A = [[0.0, 2.0], [3.0, 4.0]]  # naive elimination would divide by zero
    assert solve(A, [2.0, 7.0]) == pytest.approx([1.0, 1.0])
    assert determinant(A) == pytest.approx(-6.0)


def test_singular_matrix():
    S = [[1.0, 2.0], [2.0, 4.0]]
    assert determinant(S) == 0.0
    with pytest.raises(ValueError):
        inverse(S)
    with pytest.raises(ValueError):
        solve(S, [1.0, 1.0])


def test_rank_matches_numpy(rng):
    for _ in range(100):
        rows, cols = rng.integers(1, 6, size=2)
        r = rng.integers(1, min(rows, cols) + 1)
        A = rng.normal(size=(rows, r)) @ rng.normal(size=(r, cols))  # rank r by construction
        assert rank(A.tolist()) == np.linalg.matrix_rank(A) == r
    assert rank([[0.0, 0.0], [0.0, 0.0]]) == 0


def test_qr_properties(rng):
    for _ in range(50):
        n = rng.integers(1, 6)
        m = n + rng.integers(0, 5)
        A = rng.normal(size=(m, n))
        Q, R = qr_gram_schmidt(A)
        np.testing.assert_allclose(Q @ R, A, atol=1e-10)
        np.testing.assert_allclose(Q.T @ Q, np.eye(n), atol=1e-10)  # orthonormal columns
        np.testing.assert_allclose(R, np.triu(R))  # upper triangular
        # QR is unique once R's diagonal is positive, so it must match NumPy up to signs.
        Qn, Rn = np.linalg.qr(A)
        signs = np.sign(np.diag(Rn))
        np.testing.assert_allclose(Q, Qn * signs, atol=1e-8)


def test_least_squares_matches_numpy_lstsq(rng):
    for _ in range(50):
        A, b = rng.normal(size=(20, 4)), rng.normal(size=20)
        x = least_squares(A, b)
        np.testing.assert_allclose(x, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-8)
        # Optimality condition: the residual is orthogonal to the columns of A.
        np.testing.assert_allclose(A.T @ (A @ x - b), 0, atol=1e-8)
    R = np.triu(rng.normal(size=(4, 4))) + 3 * np.eye(4)
    y = rng.normal(size=4)
    np.testing.assert_allclose(back_substitution(R, y), np.linalg.solve(R, y))


def test_power_iteration_finds_dominant_eigenpair(rng):
    for _ in range(20):
        M = rng.normal(size=(6, 6))
        A = M @ M.T  # symmetric positive semi-definite: real eigenvalues, clear ordering
        value, vector = power_iteration(A, iters=20_000)
        eigvals, eigvecs = np.linalg.eigh(A)
        assert value == pytest.approx(eigvals[-1], rel=1e-6)
        assert abs(vector @ eigvecs[:, -1]) == pytest.approx(1.0, abs=1e-4)  # same direction
