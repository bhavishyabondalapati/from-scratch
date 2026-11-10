"""A small GAN that learns a 2D mixture of 8 Gaussians arranged on a ring (PyTorch)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def ring_of_gaussians(
    n: int, modes: int = 8, radius: float = 2.0, std: float = 0.05, generator=None
) -> torch.Tensor:
    """The real data: points around `modes` equally spaced centres on a circle."""
    k = torch.randint(0, modes, (n,), generator=generator)
    angle = 2 * math.pi * k / modes
    centres = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=1)
    return centres + std * torch.randn(n, 2, generator=generator)


def ring_centres(modes: int = 8, radius: float = 2.0) -> torch.Tensor:
    a = 2 * math.pi * torch.arange(modes) / modes
    return torch.stack([radius * torch.cos(a), radius * torch.sin(a)], dim=1)


def mlp(n_in: int, n_out: int, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, hidden), nn.LeakyReLU(0.2),
        nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
        nn.Linear(hidden, n_out),
    )  # fmt: skip


def discriminator_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """D wants real -> 1, fake -> 0:  -[log D(x) + log(1 - D(G(z)))], from logits."""
    return F.binary_cross_entropy_with_logits(
        d_real, torch.ones_like(d_real)
    ) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))


def generator_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating loss: G maximises log D(G(z)) instead of minimising log(1 - D(G(z))).
    Same fixed point, but strong gradients early on when D easily rejects the fakes."""
    return F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))


class GAN:
    def __init__(
        self, noise_dim: int = 2, hidden: int = 128, lr: float = 1e-3, seed: int = 0
    ) -> None:
        torch.manual_seed(seed)
        self.noise_dim = noise_dim
        self.G, self.D = mlp(noise_dim, 2, hidden), mlp(2, 1, hidden)
        # beta1 = 0.5 is the usual choice for GANs (less momentum, less oscillation).
        self.opt_g = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999))

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        return self.G(torch.randn(n, self.noise_dim))

    def step(self, real: torch.Tensor) -> tuple[float, float]:
        """One round of the two-player game: update D, then update G."""
        z = torch.randn(len(real), self.noise_dim)
        fake = self.G(z)
        d_loss = discriminator_loss(
            self.D(real), self.D(fake.detach())
        )  # detach: don't update G here
        self.opt_d.zero_grad()
        d_loss.backward()
        self.opt_d.step()
        g_loss = generator_loss(self.D(fake))
        self.opt_g.zero_grad()
        g_loss.backward()
        self.opt_g.step()
        return d_loss.item(), g_loss.item()

    def train(self, steps: int = 3000, batch: int = 256, seed: int = 0) -> dict[str, list[float]]:
        gen = torch.Generator().manual_seed(seed)
        history = {"d_loss": [], "g_loss": [], "modes": [], "quality": []}
        for t in range(1, steps + 1):
            d, g = self.step(ring_of_gaussians(batch, generator=gen))
            history["d_loss"].append(d)
            history["g_loss"].append(g)
            if t % 100 == 0:
                m, q = mode_metrics(self.sample(2000))
                history["modes"].append(m)
                history["quality"].append(q)
        return history


def mode_metrics(
    samples: torch.Tensor,
    modes: int = 8,
    std: float = 0.05,
    tolerance: float = 3.0,
    min_count: int = 20,
):
    """(number of modes covered, fraction of high-quality samples).

    A sample is high quality if it lies within `tolerance` standard deviations of
    some centre. A mode is covered if at least `min_count` high-quality samples
    land on it. Mode collapse shows up as few covered modes.
    """
    d = torch.cdist(samples, ring_centres(modes))
    nearest_dist, nearest = d.min(dim=1)
    good = nearest_dist < tolerance * std
    counts = torch.bincount(nearest[good], minlength=modes)
    return int((counts >= min_count).sum()), float(good.float().mean())
