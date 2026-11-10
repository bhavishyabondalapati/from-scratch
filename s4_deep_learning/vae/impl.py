"""Variational autoencoder on 8x8 digits (PyTorch)."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    """Encoder outputs a Gaussian q(z|x) = N(mu, diag(exp(logvar))) instead of one code.
    Decoder outputs Bernoulli probabilities p(x|z) for every pixel (pixels in [0, 1]).
    Prior p(z) = N(0, I), so new digits are made by decoding z ~ N(0, I).
    """

    def __init__(self, latent: int = 4, hidden: int = 128) -> None:
        super().__init__()
        self.latent = latent
        self.enc = nn.Sequential(nn.Linear(64, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, 64))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I).

        Sampling z directly is not differentiable w.r.t. mu and sigma. Moving the
        randomness into eps (which has no parameters) makes z a smooth function
        of mu and sigma, so gradients flow through.
        """
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        return self.decode_logits(self.reparameterize(mu, logvar)), mu, logvar

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(torch.randn(n, self.latent)))


def kl_to_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, 1)) summed over latent dims, per example (closed form):
    0.5 * sum(mu^2 + sigma^2 - log sigma^2 - 1)."""
    return 0.5 * (mu**2 + logvar.exp() - logvar - 1).sum(dim=1)


def negative_elbo(
    model: VAE, x: torch.Tensor, beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-ELBO = reconstruction NLL + beta * KL, per example averaged over the batch (nats).
    Returns (loss, reconstruction term, KL term)."""
    logits, mu, logvar = model(x)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)
    kl = kl_to_standard_normal(mu, logvar)
    return (recon + beta * kl).mean(), recon.mean(), kl.mean()


def train(
    model: VAE,
    X: torch.Tensor,
    epochs: int = 300,
    lr: float = 2e-3,
    batch: int = 128,
    seed: int = 0,
):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "recon": [], "kl": []}
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        sums = [0.0, 0.0, 0.0]
        for i in range(0, len(X), batch):
            x = X[perm[i : i + batch]]
            loss, recon, kl = negative_elbo(model, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            for j, v in enumerate((loss, recon, kl)):
                sums[j] += v.item() * len(x)
        for key, s in zip(history, sums):
            history[key].append(s / len(X))
    return history


@torch.no_grad()
def importance_sampled_nll(model: VAE, x: torch.Tensor, k: int = 200) -> float:
    """Tighter estimate of -log p(x) than the ELBO, using k samples from q(z|x):
    log p(x) ~= logsumexp_i [log p(x|z_i) + log p(z_i) - log q(z_i|x)] - log k."""
    mu, logvar = model.encode(x)
    std = torch.exp(0.5 * logvar)
    z = mu + std * torch.randn(k, *mu.shape)  # (k, N, L)
    log_px_z = -F.binary_cross_entropy_with_logits(
        model.decode_logits(z), x.expand(k, -1, -1), reduction="none"
    ).sum(-1)
    log_pz = (-0.5 * z**2 - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_qz = (-0.5 * ((z - mu) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_px = torch.logsumexp(log_px_z + log_pz - log_qz, dim=0) - math.log(k)
    return float(-log_px.mean())
