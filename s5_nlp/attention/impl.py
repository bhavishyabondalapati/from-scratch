"""Scaled dot-product attention and multi-head attention, written from basic tensor ops.

(PyTorch is used only for tensors and autograd; nothing here calls its built-in attention.)
"""

from __future__ import annotations

import math

import torch
from torch import nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    causal: bool = False,
    dropout: float = 0.0,
    return_weights: bool = False,
):
    """softmax(Q K^T / sqrt(d_k)) V.

    q: (..., T_q, d_k), k: (..., T_k, d_k), v: (..., T_k, d_v).
    mask: boolean, broadcastable to (..., T_q, T_k); True means "may attend".
    causal: position i may only look at positions <= i (for language models).

    Each query is compared with every key; the softmax turns the similarities
    into weights that sum to 1; the output is the weighted average of the values.
    Dividing by sqrt(d_k) keeps the dot products from growing with the dimension,
    which would push the softmax into a near one-hot region with tiny gradients.
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (..., T_q, T_k)
    if causal:
        T_q, T_k = scores.shape[-2:]
        allowed = torch.ones(T_q, T_k, dtype=torch.bool, device=q.device).tril(T_k - T_q)
        scores = scores.masked_fill(~allowed, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    weights = torch.nan_to_num(weights)  # a fully masked row gives NaN; make it all zeros
    if dropout > 0:
        weights = nn.functional.dropout(weights, dropout)
    out = weights @ v
    return (out, weights) if return_weights else out


class MultiHeadAttention(nn.Module):
    """Run h attention "heads" in parallel, each in its own d_model/h-dimensional subspace,
    then concatenate and mix them with an output projection.

    Different heads can specialise (one tracks the previous word, another the
    subject of the sentence, ...) at the same total cost as one big head.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.h, self.d_head = d_model, n_heads, d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_model) -> (B, h, T, d_head)"""
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        causal: bool = False,
        return_weights: bool = False,
    ):
        """Self-attention when key/value are omitted; cross-attention otherwise.
        mask: (B, T_k) key-padding mask or (B, T_q, T_k), True = keep."""
        key = query if key is None else key
        value = key if value is None else value
        q, k, v = (
            self._split(self.q_proj(query)),
            self._split(self.k_proj(key)),
            self._split(self.v_proj(value)),
        )
        if mask is not None:
            mask = (
                mask[:, None, None, :] if mask.dim() == 2 else mask[:, None]
            )  # broadcast over heads
        out, weights = scaled_dot_product_attention(
            q, k, v, mask, causal, self.dropout if self.training else 0.0, return_weights=True
        )
        B, _, T, _ = out.shape
        out = self.out_proj(out.transpose(1, 2).reshape(B, T, self.d_model))  # concatenate heads
        return (out, weights) if return_weights else out
