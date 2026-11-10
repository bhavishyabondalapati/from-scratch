"""Vector and matrix operations from scratch.

Part 1 uses plain Python lists so every loop is visible.
Part 2 uses NumPy arrays (for storage and basic arithmetic only) to build
bigger pieces: Gram-Schmidt QR, least squares and power iteration.
"""

from __future__ import annotations

import math

import numpy as np

Vector = list[float]
Matrix = list[list[float]]

# --- Part 1: pure Python ------------------------------------------------------


def dot(u: Vector, v: Vector) -> float:
    """u . v = sum of elementwise products = |u| |v| cos(angle)."""
    if len(u) != len(v):
        raise ValueError("vectors must have the same length")
    return sum(a * b for a, b in zip(u, v))


def norm(v: Vector, p: float = 2) -> float:
    """p-norm: p=1 Manhattan, p=2 Euclidean, p=inf max absolute value."""
    if p == math.inf:
        return max(abs(x) for x in v)
    return sum(abs(x) ** p for x in v) ** (1 / p)


def cosine_similarity(u: Vector, v: Vector) -> float:
    return dot(u, v) / (norm(u) * norm(v))


def project(u: Vector, v: Vector) -> Vector:
    """Projection of u onto v: the part of u that points along v."""
    scale = dot(u, v) / dot(v, v)
    return [scale * x for x in v]


def transpose(A: Matrix) -> Matrix:
    return [list(col) for col in zip(*A)]


def matmul(A: Matrix, B: Matrix) -> Matrix:
    """(AB)[i][j] = dot(row i of A, column j of B). O(n*m*p)."""
    if len(A[0]) != len(B):
        raise ValueError("inner dimensions do not match")
    Bt = transpose(B)
    return [[dot(row, col) for col in Bt] for row in A]


def identity(n: int) -> Matrix:
    return [[float(i == j) for j in range(n)] for i in range(n)]


def _eliminate(A: Matrix, B: Matrix) -> tuple[Matrix, Matrix, float]:
    """Gauss-Jordan elimination with partial pivoting on the augmented system [A | B].

    Returns (reduced A, transformed B, determinant of A). If A is invertible
    the reduced A is the identity, so the transformed B equals A^-1 B.
    """
    n = len(A)
    A = [row[:] for row in A]
    B = [row[:] for row in B]
    det = 1.0
    for col in range(n):
        # Partial pivoting: use the row with the largest entry in this column.
        # Dividing by a tiny pivot would blow up rounding errors.
        pivot = max(range(col, n), key=lambda r: abs(A[r][col]))
        if abs(A[pivot][col]) < 1e-12:
            return A, B, 0.0  # singular
        if pivot != col:
            A[col], A[pivot] = A[pivot], A[col]
            B[col], B[pivot] = B[pivot], B[col]
            det = -det  # a row swap flips the determinant's sign
        p = A[col][col]
        det *= p
        A[col] = [x / p for x in A[col]]
        B[col] = [x / p for x in B[col]]
        for r in range(n):
            if r != col and A[r][col] != 0:
                f = A[r][col]
                A[r] = [x - f * y for x, y in zip(A[r], A[col])]
                B[r] = [x - f * y for x, y in zip(B[r], B[col])]
    return A, B, det


def determinant(A: Matrix) -> float:
    return _eliminate(A, [[] for _ in A])[2]


def solve(A: Matrix, b: Vector) -> Vector:
    """Solve A x = b for square A."""
    _, X, det = _eliminate(A, [[x] for x in b])
    if det == 0:
        raise ValueError("matrix is singular")
    return [row[0] for row in X]


def inverse(A: Matrix) -> Matrix:
    _, inv, det = _eliminate(A, identity(len(A)))
    if det == 0:
        raise ValueError("matrix is singular")
    return inv


def rank(A: Matrix, tol: float = 1e-10) -> int:
    """Number of linearly independent rows = number of pivots in row echelon form."""
    A = [row[:] for row in A]
    rows, cols = len(A), len(A[0])
    r = 0
    for c in range(cols):
        pivot = max(range(r, rows), key=lambda i: abs(A[i][c]), default=None)
        if pivot is None or abs(A[pivot][c]) < tol:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        for i in range(r + 1, rows):
            f = A[i][c] / A[r][c]
            A[i] = [x - f * y for x, y in zip(A[i], A[r])]
        r += 1
        if r == rows:
            break
    return r


# --- Part 2: NumPy-backed --------------------------------------------------------


def qr_gram_schmidt(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """QR decomposition A = Q R via modified Gram-Schmidt (A: m x n, full column rank).

    Walk through the columns; from each one remove its components along the
    orthonormal columns found so far, then normalise what is left.
    Q has orthonormal columns; R is upper triangular.
    """
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ v  # "modified": project the CURRENT v, not the original column
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.sqrt(v @ v)
        Q[:, j] = v / R[j, j]
    return Q, R


def back_substitution(R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve R x = b for upper-triangular R, from the last row upward."""
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - R[i, i + 1 :] @ x[i + 1 :]) / R[i, i]
    return x


def least_squares(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """argmin_x ||A x - b||^2 via QR: A = QR  =>  R x = Q^T b.

    More numerically stable than the normal equations (A^T A) x = A^T b,
    because forming A^T A squares the condition number.
    """
    Q, R = qr_gram_schmidt(A)
    return back_substitution(R, Q.T @ np.asarray(b, dtype=float))


def power_iteration(
    A: np.ndarray, iters: int = 1000, tol: float = 1e-12, seed: int = 0
) -> tuple[float, np.ndarray]:
    """Largest-magnitude eigenvalue and its eigenvector.

    Repeatedly multiplying a vector by A stretches it most along the dominant
    eigenvector, so after normalising each time it converges to that direction.
    The eigenvalue is the Rayleigh quotient v^T A v.
    """
    A = np.asarray(A, dtype=float)
    v = np.random.default_rng(seed).normal(size=A.shape[0])
    v /= np.sqrt(v @ v)
    eigenvalue = 0.0
    for _ in range(iters):
        w = A @ v
        w /= np.sqrt(w @ w)
        new_eigenvalue = w @ A @ w
        converged = abs(new_eigenvalue - eigenvalue) < tol
        v, eigenvalue = w, new_eigenvalue
        if converged:
            break
    return float(eigenvalue), v
