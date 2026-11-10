"""Positional encodings: sinusoidal (absolute) and rotary (RoPE, relative)."""

from __future__ import annotations

import math

import torch


def sinusoidal_encoding(max_len: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """PE[pos, 2i] = sin(pos / base^(2i/d)),  PE[pos, 2i+1] = cos(pos / base^(2i/d)).

    Every pair of dimensions is a clock hand turning at its own speed: the first
    pairs spin fast (fine position), the last ones very slowly (coarse position).
    Added to the token embeddings so attention can tell positions apart.
    """
    pos = torch.arange(max_len, dtype=torch.float64)[:, None]
    freqs = base ** (-torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)
    pe = torch.zeros(max_len, d_model, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * freqs)
    pe[:, 1::2] = torch.cos(pos * freqs)
    return pe.float()


def rope_frequencies(
    d_head: int, max_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (max_len, d_head // 2) for rotary embeddings."""
    if d_head % 2:
        raise ValueError("RoPE needs an even head dimension")
    inv_freq = base ** (-torch.arange(0, d_head, 2, dtype=torch.float64) / d_head)
    angles = torch.arange(max_len, dtype=torch.float64)[:, None] * inv_freq[None, :]
    return torch.cos(angles).float(), torch.sin(angles).float()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, offset: int = 0
) -> torch.Tensor:
    """Rotate each consecutive pair (x[2i], x[2i+1]) of a query/key vector by angle pos * theta_i.

    x: (..., T, d_head). Applied to queries AND keys (not values). Because a
    rotation by a then the inverse rotation by b is a rotation by (a - b), the
    dot product q_m . k_n depends only on the RELATIVE offset m - n.
    `offset` is the position of the first token (used with a KV cache).
    """
    T = x.shape[-2]
    c, s = cos[offset : offset + T].to(x.dtype), sin[offset : offset + T].to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


def rope_as_complex(
    x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0
) -> torch.Tensor:
    """Reference: the same rotation written as multiplying complex numbers by e^{i pos theta}."""
    d = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    z = torch.complex(x[..., 0::2].double(), x[..., 1::2].double())
    rot = torch.polar(
        torch.ones(len(positions), d // 2, dtype=torch.float64),
        positions[:, None].double() * inv_freq,
    )
    out = z * rot
    return torch.stack([out.real, out.imag], dim=-1).flatten(-2).to(x.dtype)


def wavelengths(d_model: int, base: float = 10000.0) -> torch.Tensor:
    """Period (in tokens) of each sin/cos pair: 2 pi to 2 pi * base."""
    return 2 * math.pi * base ** (torch.arange(0, d_model, 2, dtype=torch.float64) / d_model)
