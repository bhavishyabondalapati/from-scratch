"""Autoencoder: compress 64-pixel digits to a small code and reconstruct them (PyTorch)."""

from __future__ import annotations

import torch
from torch import nn


class Autoencoder(nn.Module):
    """encoder: 64 -> hidden -> code ;  decoder: code -> hidden -> 64 (sigmoid, pixels in [0, 1]).

    With linear=True there are no hidden layers or non-linearities: a linear
    autoencoder trained with squared error learns the same subspace as PCA.
    """

    def __init__(self, code_dim: int = 8, hidden: int = 128, linear: bool = False) -> None:
        super().__init__()
        if linear:
            self.encoder = nn.Linear(64, code_dim)
            self.decoder = nn.Linear(code_dim, 64)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(64, hidden), nn.ReLU(), nn.Linear(hidden, code_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(code_dim, hidden), nn.ReLU(), nn.Linear(hidden, 64), nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train(
    model: Autoencoder,
    X: torch.Tensor,
    epochs: int = 200,
    lr: float = 3e-3,
    batch: int = 128,
    noise: float = 0.0,
    seed: int = 0,
) -> list[float]:
    """Minimise mean squared reconstruction error. noise > 0 gives a DENOISING
    autoencoder: corrupt the input, but ask for the clean image back."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), batch):
            x = X[perm[i : i + batch]]
            x_in = x + noise * torch.randn_like(x) if noise else x
            loss = ((model(x_in) - x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        history.append(total / len(X))
    return history


@torch.no_grad()
def reconstruction_mse(model: Autoencoder, X: torch.Tensor) -> float:
    return float(((model(X) - X) ** 2).mean())
