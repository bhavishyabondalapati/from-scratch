"""ResNet basic block and a small residual network vs an equally deep plain network (PyTorch)."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """out = ReLU( F(x) + shortcut(x) ),  F = conv3x3 -> BN -> ReLU -> conv3x3 -> BN.

    The block only has to learn the RESIDUAL F(x) = desired(x) - x. If the best
    thing to do is nothing, it can push F towards zero instead of having to learn
    an exact identity map through two convolutions.

    When the block changes the number of channels or downsamples (stride 2), the
    shortcut becomes a 1x1 conv (+ BN) so the shapes match before adding.
    """

    def __init__(self, c_in: int, c_out: int, stride: int = 1, residual: bool = True) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)  # BN makes the bias redundant
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return torch.relu(out)


class SmallResNet(nn.Module):
    """Stem conv -> `n_blocks` blocks at width 16 -> one downsampling block to 32 ->
    global average pool -> linear. With residual=False the same layers form a plain net."""

    def __init__(self, n_blocks: int = 8, n_classes: int = 10, residual: bool = True) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16), nn.ReLU()
        )
        blocks = [BasicBlock(16, 16, residual=residual) for _ in range(n_blocks)]
        blocks.append(BasicBlock(16, 32, stride=2, residual=residual))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        return self.head(x.mean(dim=(2, 3)))  # global average pooling


def train(
    model: nn.Module,
    X,
    y,
    epochs: int = 10,
    lr: float = 0.05,
    batch: int = 64,
    seed: int = 0,
    device="cpu",
):
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        total = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i : i + batch]
            xb, yb = X[idx].to(device), y[idx].to(device)
            loss = nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history.append(total / len(X))
    model.eval()
    return history


@torch.no_grad()
def accuracy(model: nn.Module, X, y, device="cpu") -> float:
    model.eval()
    return float((model(X.to(device)).argmax(1).cpu() == y).float().mean())


def first_layer_grad_norm(model: SmallResNet, X, y) -> float:
    """Size of the gradient reaching the stem conv at initialisation."""
    model.train()
    model.zero_grad()
    nn.functional.cross_entropy(model(X), y).backward()
    return float(model.stem[0].weight.grad.norm())
