"""A small decoder-only transformer (GPT-style) for character-level language modelling.

Built from this repo's own attention (s5_nlp/attention) and rotary position
embeddings (s5_nlp/positional_encoding). Supports a KV cache for fast generation
and greedy / temperature / top-k / top-p sampling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from s5_nlp.attention.impl import scaled_dot_product_attention
from s5_nlp.positional_encoding.impl import apply_rope, rope_frequencies


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128  # maximum context length
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.h, self.d_head = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout
        cos, sin = rope_frequencies(self.d_head, cfg.block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        """x: (B, T, d). If `cache` is given, past keys/values are read from it and
        the new ones appended, so each new token costs O(T) instead of O(T^2)."""
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.view(B, T, self.h, self.d_head).transpose(1, 2) for t in (q, k, v))
        past = 0 if cache is None or "k" not in cache else cache["k"].shape[2]
        q = apply_rope(q, self.cos, self.sin, offset=past)  # positions past .. past+T-1
        k = apply_rope(k, self.cos, self.sin, offset=past)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # causal=True aligns the mask to the END, so new queries see all cached keys.
        y = scaled_dot_product_attention(
            q, k, v, causal=True, dropout=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), cache))
        return x + self.drop(self.mlp(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying: input and output embeddings shared
        self.apply(self._init)
        for name, p in self.named_parameters():  # GPT-2: scale residual projections by depth
            if name.endswith("proj.weight") or name.endswith("mlp.2.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        caches: list[dict] | None = None,
    ):
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i])
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- Sampling ------------------------------------------------------------------------


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k largest logits per row, set the rest to -inf."""
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def filter_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose probabilities add up to >= p."""
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    # Drop a token if the tokens BEFORE it already reach p (so the top token always stays).
    drop_sorted = (probs.cumsum(-1) - probs) >= p
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator=None,
) -> torch.Tensor:
    """logits: (B, V) -> next token ids (B,). temperature 0 means greedy (argmax)."""
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k is not None:
        logits = filter_top_k(logits, top_k)
    if top_p is not None:
        logits = filter_top_p(logits, top_p)
    return torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).squeeze(-1)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    n_new: int,
    use_cache: bool = True,
    seed: int | None = None,
    **sampling,
) -> torch.Tensor:
    """Append n_new tokens to idx (B, T). With the KV cache only the newest token is fed
    through the network at each step. (The context must stay within block_size.)"""
    model.eval()
    gen = None if seed is None else torch.Generator(device=idx.device).manual_seed(seed)
    if idx.shape[1] + n_new > model.cfg.block_size:
        raise ValueError("prompt + new tokens exceed block_size")
    if not use_cache:
        for _ in range(n_new):
            logits, _ = model(idx)
            idx = torch.cat(
                [idx, sample_next(logits[:, -1], generator=gen, **sampling)[:, None]], dim=1
            )
        return idx
    caches = [{} for _ in model.blocks]
    logits, _ = model(idx, caches=caches)  # "prefill": process the whole prompt once
    for _ in range(n_new):
        nxt = sample_next(logits[:, -1], generator=gen, **sampling)[:, None]
        idx = torch.cat([idx, nxt], dim=1)
        logits, _ = model(nxt, caches=caches)  # "decode": one token at a time
    return idx


# --- Data and training ------------------------------------------------------------------------


class CharTokenizer:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)


def get_batch(data: torch.Tensor, block: int, batch: int, generator: torch.Generator, device="cpu"):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=generator)
    x = torch.stack([data[i : i + block] for i in ix])
    y = torch.stack([data[i + 1 : i + block + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(
    model: GPT, data: torch.Tensor, block: int, batch: int = 64, iters: int = 20, device="cpu"
) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    losses = [model(*get_batch(data, block, batch, gen, device))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def train(
    model: GPT,
    train_data,
    val_data,
    steps: int = 2000,
    batch: int = 64,
    lr: float = 1e-3,
    warmup: int = 100,
    eval_every: int = 250,
    device="cpu",
    seed: int = 0,
):
    """AdamW with linear warm-up then cosine decay; gradient clipping at 1.0."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(device).train()
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
    )

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        return 0.1 * lr + 0.45 * lr * (
            1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))
        )

    history = {"step": [], "train": [], "val": []}
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = get_batch(train_data, model.cfg.block_size, batch, gen, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            history["step"].append(step + 1)
            history["train"].append(
                estimate_loss(model, train_data, model.cfg.block_size, device=device)
            )
            history["val"].append(
                estimate_loss(model, val_data, model.cfg.block_size, device=device)
            )
    model.eval()
    return history


def shakespeare_splits():
    from data_utils import tiny_shakespeare

    text = tiny_shakespeare()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    n = int(0.9 * len(data))
    return tok, data[:n], data[n:]


CHECKPOINT = Path(__file__).resolve().parents[2] / "data" / "gpt_char_shakespeare.pt"


def pretrained_shakespeare_gpt(steps: int = 2500, device=None):
    """Train once (a few minutes on Apple mps) and cache the weights under data/ (gitignored)."""
    from s4_deep_learning.torch_utils import get_device

    tok, train_data, val_data = shakespeare_splits()
    cfg = GPTConfig(vocab_size=len(tok.chars))
    model = GPT(cfg)
    if CHECKPOINT.exists():
        saved = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(saved["model"])
        return model.eval(), tok, saved["history"]
    device = device or get_device()
    history = train(model, train_data, val_data, steps=steps, device=device)
    model.to("cpu")
    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "history": history}, CHECKPOINT)
    return model.eval(), tok, history
