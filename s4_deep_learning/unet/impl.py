"""U-Net for semantic segmentation of a synthetic shapes dataset (PyTorch).

Task: 32x32 grey images contain several discs and several squares of the SAME
brightness, plus noise. The model must mark the pixels belonging to discs only.
Because brightness alone cannot tell discs from squares, a pixel's label depends
on the shape around it: the network needs context (downsampling) AND precise
locations (skip connections).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def make_shapes(n: int, size: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns images (n, 1, size, size) and binary disc masks (n, 1, size, size)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    images = np.zeros((n, size, size), dtype=np.float32)
    masks = np.zeros((n, size, size), dtype=np.float32)
    for i in range(n):
        for _ in range(rng.integers(1, 4)):  # discs (the target class)
            r = rng.uniform(3, 6)
            cy, cx = rng.uniform(r, size - r, size=2)
            disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
            images[i][disc] = 1.0
            masks[i][disc] = 1.0
        for _ in range(rng.integers(1, 4)):  # squares (distractors)
            s = int(rng.integers(5, 11))
            y0, x0 = rng.integers(0, size - s, size=2)
            images[i, y0 : y0 + s, x0 : x0 + s] = 1.0
            masks[i, y0 : y0 + s, x0 : x0 + s] = 0.0  # a square drawn on top hides the disc
    images += rng.normal(0, 0.2, images.shape).astype(np.float32)
    return torch.tensor(images)[:, None], torch.tensor(masks)[:, None]


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
    )  # fmt: skip


class UNet(nn.Module):
    """Encoder: conv blocks + max-pool (32 -> 16 -> 8), widening channels.
    Decoder: transposed conv upsampling (8 -> 16 -> 32); at each level the
    encoder's feature map is CONCATENATED back in (the skip connection), giving
    the decoder the fine spatial detail that pooling threw away.
    With skips=False the decoder only sees the upsampled bottleneck.
    """

    def __init__(self, base: int = 16, skips: bool = True) -> None:
        super().__init__()
        self.skips = skips
        b = base
        self.enc1, self.enc2 = conv_block(1, b), conv_block(b, 2 * b)
        self.bottleneck = conv_block(2 * b, 4 * b)
        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, stride=2)
        self.dec2 = conv_block(4 * b if skips else 2 * b, 2 * b)
        self.up1 = nn.ConvTranspose2d(2 * b, b, 2, stride=2)
        self.dec1 = conv_block(2 * b if skips else b, b)
        self.head = nn.Conv2d(b, 1, 1)  # per-pixel logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)  # 32x32
        e2 = self.enc2(F.max_pool2d(e1, 2))  # 16x16
        z = self.bottleneck(F.max_pool2d(e2, 2))  # 8x8
        d2 = self.up2(z)
        d2 = self.dec2(torch.cat([d2, e2], dim=1) if self.skips else d2)
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1) if self.skips else d1)
        return self.head(d1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice. Dice = 2|A ∩ B| / (|A| + |B|), computed softly on probabilities.
    Insensitive to how much background there is, unlike pixel-wise accuracy."""
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()


def iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Intersection over union of two binary masks, pooled over the whole set."""
    pred, target = pred.bool(), target.bool()
    return float((pred & target).sum() / (pred | target).sum().clamp(min=1))


def train(
    model: UNet,
    X,
    Y,
    epochs: int = 25,
    lr: float = 3e-3,
    batch: int = 32,
    seed: int = 0,
    device="cpu",
):
    """BCE + Dice loss with Adam. Pass device="mps" on Apple silicon (~20x faster here)."""
    torch.manual_seed(seed)
    model.to(device)
    X, Y = X.to(device), Y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X)).to(device)
        total = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i : i + batch]
            logits = model(X[idx])
            loss = F.binary_cross_entropy_with_logits(logits, Y[idx]) + dice_loss(logits, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history.append(total / len(X))
    model.eval()
    return history


@torch.no_grad()
def predict(model: UNet, X) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    return (model(X.to(device)) > 0).float().cpu()
