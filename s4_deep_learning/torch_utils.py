"""Small helpers shared by the PyTorch folders of Section 4."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_digits


def get_device(prefer_mps: bool = True) -> torch.device:
    """Apple-silicon GPU ("mps") when available, otherwise CPU.

    The tests pass device="cpu" for exact reproducibility; small models are often
    just as fast on CPU anyway.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def digits_tensors(seed: int = 0, n_train: int = 1400):
    """sklearn's 8x8 digits as float tensors in [0, 1], shape (N, 1, 8, 8), split train/test."""
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X / 16.0, dtype=torch.float32).reshape(-1, 1, 8, 8)
    y = torch.tensor(y)
    idx = torch.randperm(len(y), generator=torch.Generator().manual_seed(seed))
    tr, te = idx[:n_train], idx[n_train:]
    return X[tr], y[tr], X[te], y[te]


def save_curves(
    path: str | Path,
    curves: dict[str, list[float]],
    title: str,
    ylabel: str = "loss",
    logy: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    for name, values in curves.items():
        ax.plot(values, label=name)
    ax.set(xlabel="epoch", ylabel=ylabel, title=title)
    if logy:
        ax.set_yscale("log")
    if len(curves) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
