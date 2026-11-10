"""Siamese network with contrastive loss: learn an embedding where same-class images are close.

Trained on pairs of digits 0-6. Evaluated on new images of those classes, and
on 7, 8, 9, classes it never saw (the test of whether the embedding transfers).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class EmbeddingNet(nn.Module):
    """Small CNN mapping an 8x8 digit to a vector. BOTH images of a pair go
    through the SAME network (shared weights): that is what "Siamese" means."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 2 * 2, 128), nn.ReLU(), nn.Linear(128, dim),
        )  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def contrastive_loss(
    e1: torch.Tensor, e2: torch.Tensor, same: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Hadsell et al. (2006):
        same pair:      d^2                     (pull together)
        different pair: max(0, margin - d)^2    (push apart, but only until `margin`)
    The margin stops the network from wasting effort pushing already-distant pairs."""
    d = F.pairwise_distance(e1, e2)
    same = same.float()
    return (same * d**2 + (1 - same) * F.relu(margin - d) ** 2).mean() / 2


def make_pairs(X: torch.Tensor, y: torch.Tensor, n: int, generator: torch.Generator):
    """n random pairs, half from the same class and half from different classes (vectorised)."""
    i1 = torch.randint(len(y), (n,), generator=generator)
    same = torch.arange(n) % 2 == 0
    # For each anchor draw candidate partners until each has the required relation.
    i2 = torch.randint(len(y), (n,), generator=generator)
    wrong = (y[i1] == y[i2]) != same
    while wrong.any():
        i2[wrong] = torch.randint(len(y), (int(wrong.sum()),), generator=generator)
        wrong = (y[i1] == y[i2]) != same
    return X[i1], X[i2], same


def train(
    model: EmbeddingNet,
    X,
    y,
    steps: int = 1500,
    batch: int = 128,
    lr: float = 1e-3,
    margin: float = 1.0,
    seed: int = 0,
):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        a, b, same = make_pairs(X, y, batch, gen)
        loss = contrastive_loss(model(a), model(b), same, margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return history


@torch.no_grad()
def pair_distances(model: EmbeddingNet, a, b) -> torch.Tensor:
    model.eval()
    return F.pairwise_distance(model(a), model(b))
