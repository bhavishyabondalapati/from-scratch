"""LoRA (low-rank adaptation), implemented by hand on nn.Linear and applied to the tiny GPT."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """y = x W^T + b  +  (alpha / r) * x A^T B^T

    W (d_out x d_in) is FROZEN. Only A (r x d_in) and B (d_out x r) are trained,
    so the weight update is restricted to the low-rank matrix  delta_W = (alpha/r) B A.
    B starts at zero, so at initialisation the layer behaves exactly like the original.
    Trainable parameters: r * (d_in + d_out) instead of d_in * d_out.
    """

    def __init__(
        self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same init as nn.Linear's weight
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        return self.scale * self.B @ self.A

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not self.merged:
            out = out + self.scale * (self.dropout(x) @ self.A.T) @ self.B.T
        return out

    @torch.no_grad()
    def merge(self) -> None:
        """Fold delta_W into W for inference: zero extra latency, same outputs."""
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False


def apply_lora(
    model: nn.Module,
    target_names: tuple[str, ...] = ("qkv", "proj"),
    r: int = 4,
    alpha: float = 8.0,
) -> nn.Module:
    """Freeze every parameter, then wrap each nn.Linear whose attribute name is in
    `target_names` with a LoRALinear (only those adapters are trainable)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if name in target_names and isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha))
    return model


def merge_lora(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def finetune_lm(
    model,
    data: torch.Tensor,
    steps: int,
    lr: float,
    batch: int = 32,
    block: int = 64,
    seed: int = 0,
    device="cpu",
) -> list[float]:
    """Next-character training on `data`, updating only parameters with requires_grad."""
    from s5_nlp.gpt.impl import get_batch

    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(device).train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    history = []
    for _ in range(steps):
        x, y = get_batch(data, block, batch, gen, device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    model.eval()
    return history
