"""Sequence-to-sequence GRU encoder-decoder with Bahdanau (additive) attention (PyTorch).

Toy task: reverse a random string of letters ("abcd" -> "dcba"). The output's
first token depends on the input's LAST token, which is the hardest case for a
model that must squeeze the whole input into one vector.
"""

from __future__ import annotations

import torch
from torch import nn

PAD, BOS, EOS = 0, 1, 2
LETTERS = "abcdefghijklmnopqrstuvwxyz"[:10]
VOCAB = ["<pad>", "<s>", "</s>", *LETTERS]


def encode(s: str) -> list[int]:
    return [3 + LETTERS.index(c) for c in s]


def decode(ids: list[int]) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        if i >= 3:
            out.append(LETTERS[i - 3])
    return "".join(out)


def make_batch(n: int, min_len: int, max_len: int, generator: torch.Generator):
    """Random strings and their reversals, padded. Targets end with </s>."""
    lengths = torch.randint(min_len, max_len + 1, (n,), generator=generator)
    src = torch.full((n, max_len), PAD)
    tgt = torch.full((n, max_len + 1), PAD)
    for i, L in enumerate(lengths.tolist()):
        tokens = torch.randint(3, 3 + len(LETTERS), (L,), generator=generator)
        src[i, :L] = tokens
        tgt[i, :L] = tokens.flip(0)
        tgt[i, L] = EOS
    return src, lengths, tgt


class AdditiveAttention(nn.Module):
    """Bahdanau attention: score(s, h_j) = v . tanh(W_s s + W_h h_j).

    A tiny neural network decides how relevant each encoder state h_j is to the
    current decoder state s. (Luong / transformer attention uses a dot product instead.)
    """

    def __init__(self, dec_dim: int, enc_dim: int, attn_dim: int) -> None:
        super().__init__()
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(enc_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, s: torch.Tensor, H: torch.Tensor, mask: torch.Tensor):
        """s: (B, dec_dim), H: (B, T, enc_dim), mask: (B, T) True for real tokens.
        Returns context (B, enc_dim) and weights (B, T)."""
        scores = self.v(torch.tanh(self.W_s(s)[:, None, :] + self.W_h(H))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return (weights[:, :, None] * H).sum(dim=1), weights


class Seq2Seq(nn.Module):
    def __init__(
        self, vocab: int = len(VOCAB), emb: int = 32, hidden: int = 64, attention: bool = True
    ) -> None:
        super().__init__()
        self.attention = attention
        self.embed = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.encoder = nn.GRU(emb, hidden, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hidden, hidden)  # encoder summary -> initial decoder state
        self.attn = AdditiveAttention(hidden, 2 * hidden, hidden)
        ctx_dim = 2 * hidden if attention else 0
        self.decoder = nn.GRUCell(emb + ctx_dim, hidden)
        self.out = nn.Linear(hidden + ctx_dim, vocab)

    def encode(self, src: torch.Tensor, lengths: torch.Tensor):
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embed(src), lengths, batch_first=True, enforce_sorted=False
        )
        H, h_n = self.encoder(packed)
        H, _ = nn.utils.rnn.pad_packed_sequence(H, batch_first=True, total_length=src.shape[1])
        summary = torch.cat([h_n[0], h_n[1]], dim=-1)  # last forward + last backward state
        mask = torch.arange(src.shape[1])[None, :] < lengths[:, None]
        return H, torch.tanh(self.bridge(summary)), mask

    def step(self, y_prev, s, H, mask):
        """One decoder step: (optionally) attend, update state, predict next-token logits."""
        e = self.embed(y_prev)
        if self.attention:
            ctx, w = self.attn(s, H, mask)
            s = self.decoder(torch.cat([e, ctx], dim=-1), s)
            return self.out(torch.cat([s, ctx], dim=-1)), s, w
        s = self.decoder(e, s)
        return self.out(s), s, None

    def forward(self, src, lengths, tgt):
        """Teacher forcing: feed the TRUE previous token at every step. Returns logits (B, T_out, V)."""
        H, s, mask = self.encode(src, lengths)
        y_prev = torch.full((len(src),), BOS)
        logits = []
        for t in range(tgt.shape[1]):
            lg, s, _ = self.step(y_prev, s, H, mask)
            logits.append(lg)
            y_prev = tgt[:, t]
        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def greedy_decode(self, src, lengths, max_len: int):
        """Feed the model's OWN previous prediction. Returns token ids and attention maps."""
        H, s, mask = self.encode(src, lengths)
        y = torch.full((len(src),), BOS)
        tokens, maps = [], []
        for _ in range(max_len):
            lg, s, w = self.step(y, s, H, mask)
            y = lg.argmax(-1)
            tokens.append(y)
            maps.append(w)
        attn = torch.stack(maps, dim=1) if self.attention else None
        return torch.stack(tokens, dim=1), attn


def train(
    model: Seq2Seq,
    steps: int = 3000,
    batch: int = 64,
    min_len: int = 3,
    max_len: int = 12,
    lr: float = 2e-3,
    seed: int = 0,
):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(steps):
        src, lengths, tgt = make_batch(batch, min_len, max_len, gen)
        logits = model(src, lengths, tgt)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=PAD
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(loss.item())
    return history


def exact_match(model: Seq2Seq, n: int, length: int, seed: int = 123) -> float:
    """Fraction of test strings of a given length reversed perfectly."""
    model.eval()
    src, lengths, tgt = make_batch(n, length, length, torch.Generator().manual_seed(seed))
    pred, _ = model.greedy_decode(src, lengths, length + 1)
    return float(sum(decode(p.tolist()) == decode(t.tolist()) for p, t in zip(pred, tgt)) / n)
