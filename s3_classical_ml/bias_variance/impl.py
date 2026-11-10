"""Bias-variance decomposition demo and a regularisation demo, using polynomial regression."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from s3_classical_ml.ridge_lasso.impl import Ridge


def true_function(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """[x, x^2, ..., x^degree] (the intercept is handled by the model)."""
    return np.column_stack([x**p for p in range(1, degree + 1)])


def sample_dataset(
    rng: np.random.Generator, n: int = 30, noise: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(0, 1, size=n)
    return x, true_function(x) + noise * rng.normal(size=n)


def bias_variance_decomposition(
    degree: int,
    alpha: float = 1e-8,
    n_datasets: int = 200,
    n_train: int = 30,
    noise: float = 0.3,
    seed: int = 0,
) -> dict[str, float]:
    """Estimate bias^2, variance and noise of a model class by Monte Carlo.

    Train the same kind of model on many independent training sets. At fixed
    test inputs x:
      bias^2   = (average prediction - true f(x))^2       -- systematic error
      variance = spread of predictions across datasets    -- sensitivity to the sample
      noise    = irreducible variance of y around f(x)
    Expected squared error on a fresh (x, y) = bias^2 + variance + noise.
    """
    rng = np.random.default_rng(seed)
    x_test = np.linspace(0.02, 0.98, 50)
    preds = np.empty((n_datasets, len(x_test)))
    for d in range(n_datasets):
        x, y = sample_dataset(rng, n_train, noise)
        model = Ridge(alpha).fit(polynomial_features(x, degree), y)
        preds[d] = model.predict(polynomial_features(x_test, degree))
    f = true_function(x_test)
    mean_pred = preds.mean(axis=0)
    bias_sq = float(np.mean((mean_pred - f) ** 2))
    variance = float(np.mean(preds.var(axis=0)))
    # Direct estimate of expected test error, using fresh noisy labels at x_test.
    y_test = f + noise * rng.normal(size=(n_datasets, len(x_test)))
    test_mse = float(np.mean((preds - y_test) ** 2))
    return {"bias_sq": bias_sq, "variance": variance, "noise": noise**2, "test_mse": test_mse}


def regularization_path(
    degree: int = 12, alphas: np.ndarray | None = None, seed: int = 2
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train and validation MSE of a high-degree ridge model across a range of alphas.

    Small alpha overfits (low train, high val); large alpha underfits (both high);
    validation error is U-shaped and its minimum picks alpha.
    Returns (alphas, train_mse, val_mse, weight norms).
    """
    alphas = np.logspace(-10, 2, 25) if alphas is None else alphas
    rng = np.random.default_rng(seed)
    x_tr, y_tr = sample_dataset(rng, 20)
    x_va, y_va = sample_dataset(rng, 500)
    # Standardise features (fitted on train) so a single alpha treats all powers fairly.
    F_tr = polynomial_features(x_tr, degree)
    mu, sd = F_tr.mean(0), F_tr.std(0)
    F_tr = (F_tr - mu) / sd
    F_va = (polynomial_features(x_va, degree) - mu) / sd
    train, val, norms = [], [], []
    for a in alphas:
        m = Ridge(a).fit(F_tr, y_tr)
        train.append(np.mean((m.predict(F_tr) - y_tr) ** 2))
        val.append(np.mean((m.predict(F_va) - y_va) ** 2))
        norms.append(np.linalg.norm(m.coef_))
    return alphas, np.array(train), np.array(val), np.array(norms)


def save_plots(out_dir: str | Path) -> None:
    """Write bias_variance.png and regularization.png into out_dir."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    degrees = list(range(1, 13))
    parts = [bias_variance_decomposition(d) for d in degrees]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in [("bias_sq", "bias²"), ("variance", "variance"), ("test_mse", "test MSE")]:
        ax.plot(degrees, [p[key] for p in parts], marker="o", label=label)
    ax.axhline(parts[0]["noise"], ls="--", c="gray", label="noise")
    ax.set(
        xlabel="polynomial degree", ylabel="error", yscale="log", title="Bias-variance trade-off"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "bias_variance.png", dpi=120)
    alphas, tr, va, _ = regularization_path()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(alphas, tr, marker="o", label="train")
    ax.loglog(alphas, va, marker="o", label="validation")
    ax.set(xlabel="ridge alpha", ylabel="MSE", title="Regularisation (degree-12 polynomial)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "regularization.png", dpi=120)
    plt.close("all")


if __name__ == "__main__":
    save_plots(Path(__file__).parent)
