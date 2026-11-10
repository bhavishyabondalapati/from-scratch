# Vector and matrix operations

## Intuition
- A **vector** is a list of numbers: a point, or an arrow from the origin.
- The **dot product** measures how much two arrows point the same way. Zero means perpendicular.
- A **matrix** is a function that transforms vectors (rotate, stretch, project). Multiplying
  matrices composes the functions.
- **Solving `Ax = b`** asks: which input does the transformation send to `b`?
- An **eigenvector** is a direction the matrix only stretches (does not rotate); the
  **eigenvalue** is the stretch factor.

Almost all of ML is these operations: a linear layer is `Wx + b`, similarity search is a dot
product, linear regression is least squares, PCA is an eigen-decomposition.

## Core math
- `u · v = Σ uᵢvᵢ = ‖u‖‖v‖cos θ`; cosine similarity = `u · v / (‖u‖‖v‖)`.
- Norms: `‖v‖₁ = Σ|vᵢ|`, `‖v‖₂ = √Σvᵢ²`, `‖v‖∞ = max|vᵢ|`.
- Projection of u on v: `(u·v / v·v) v`.
- `(AB)ᵢⱼ = Σₖ AᵢₖBₖⱼ`; `(AB)ᵀ = BᵀAᵀ`; `(AB)⁻¹ = B⁻¹A⁻¹`.
- Gaussian elimination: row operations reduce `[A | b]` to `[I | x]`. The determinant is the
  product of pivots (sign flipped per row swap).
- QR: `A = QR`, `QᵀQ = I`, `R` upper triangular. Least squares: `Rx = Qᵀb`.
- Normal equations: `AᵀA x = Aᵀb` (same solution, worse conditioning).
- Power iteration: `v ← Av / ‖Av‖` converges at rate `|λ₂/λ₁|` per step.

## Complexity
| Operation | Time |
|---|---|
| dot, norm | O(n) |
| matmul (n×m by m×p) | O(n·m·p); O(n³) for square |
| solve / inverse / determinant / rank (elimination) | O(n³) |
| QR, least squares (m×n) | O(m·n²) |
| power iteration | O(n²) per step |

## Common pitfalls
- Computing `inverse(A) @ b` instead of `solve(A, b)`: slower and less accurate.
- No pivoting: a zero or tiny pivot ruins elimination even for well-behaved matrices.
- Comparing floats with `==`; use a tolerance. "Singular" in floating point is a judgment call.
- Classical (unmodified) Gram-Schmidt loses orthogonality on ill-conditioned inputs.
- Shape bugs: (n,) vs (n,1) vs (1,n); matrix multiplication is not commutative.
- Power iteration stalls when the top two eigenvalues have (nearly) equal magnitude.

## When to use vs. alternatives
- Square system → LU/elimination. Over-determined system → QR least squares (or SVD if
  rank-deficient). Symmetric positive-definite → Cholesky (half the work).
- All eigenvalues → QR algorithm (`numpy.linalg.eigh`). Just the top one or huge sparse
  matrices → power iteration / Lanczos (this is how PageRank works).
- In real code always use NumPy/BLAS: same math, 100–1000× faster.

## Interview questions
1. **Geometric meaning of the dot product?** `‖u‖‖v‖cos θ`: projection length times length;
   zero iff orthogonal.
2. **Why not invert a matrix to solve a linear system?** It costs more and amplifies rounding
   error; factorise and substitute instead.
3. **What does the determinant tell you?** The volume scaling factor of the transformation; zero
   means the matrix squashes space into a lower dimension and is not invertible.
4. **Why is QR preferred to the normal equations?** `cond(AᵀA) = cond(A)²`, so the normal
   equations lose twice as many digits.
5. **What is the rank and why does it matter in ML?** The number of independent directions in
   the data; low rank means redundant features (collinearity) and enables compression (LoRA, PCA).
