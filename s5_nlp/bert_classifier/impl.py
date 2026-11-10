"""A tiny BERT-style encoder: bidirectional transformer, [CLS] classification, masked-LM objective.

Dataset: generated movie-review sentences whose sentiment depends on word ORDER
(negation, intensifiers, and "but" clauses), so a bag-of-words model cannot solve it.
"""

from __future__ import annotations

import random

import torch
from torch import nn
from torch.nn import functional as F

from s5_nlp.attention.impl import MultiHeadAttention

# --- Data ----------------------------------------------------------------------------

POSITIVE = ["good", "great", "wonderful", "excellent", "fun", "moving"]
NEGATIVE = ["bad", "awful", "boring", "terrible", "dull", "weak"]
SUBJECTS = [
    "the movie",
    "the plot",
    "the acting",
    "the film",
    "the ending",
    "the music",
    "the cast",
]
INTENSIFIERS = ["", "", "very", "really", "quite"]
SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
PAD, CLS, SEP, MASK, UNK = range(5)


def make_clause(rnd: random.Random) -> tuple[str, int]:
    """'the plot was not very good' -> (text, 1 if positive else 0)."""
    positive = rnd.random() < 0.5
    adjective = rnd.choice(POSITIVE if positive else NEGATIVE)
    negated = rnd.random() < 0.4
    words = [
        rnd.choice(SUBJECTS),
        "was",
        "not" if negated else "",
        rnd.choice(INTENSIFIERS),
        adjective,
    ]
    return " ".join(w for w in words if w), int(positive != negated)


def make_review(rnd: random.Random) -> tuple[str, int]:
    """One clause, or 'A but B' where the clause after 'but' decides the label."""
    first, label = make_clause(rnd)
    if rnd.random() < 0.5:
        second, label = make_clause(rnd)
        return f"{first} but {second}", label
    return first, label


def make_dataset(n: int, seed: int) -> tuple[list[str], list[int]]:
    rnd = random.Random(seed)
    pairs = [make_review(rnd) for _ in range(n)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


class WordTokenizer:
    def __init__(self, texts: list[str]) -> None:
        words = sorted({w for t in texts for w in t.split()})
        self.itos = SPECIALS + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode_batch(
        self, texts: list[str], max_len: int = 24
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[CLS] words [SEP] [PAD]...  -> ids (B, max_len) and attention mask (True = real token)."""
        ids = torch.full((len(texts), max_len), PAD)
        for i, t in enumerate(texts):
            toks = [CLS] + [self.stoi.get(w, UNK) for w in t.split()][: max_len - 2] + [SEP]
            ids[i, : len(toks)] = torch.tensor(toks)
        return ids, ids != PAD


# --- Model ------------------------------------------------------------------------------


class EncoderLayer(nn.Module):
    """Post-norm block like the original BERT: LN(x + Attn(x)), then LN(x + FFN(x)).
    No causal mask: every token sees the whole sentence (left AND right context)."""

    def __init__(self, d: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d, heads, dropout)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.drop(self.attn(x, mask=mask)))  # padding keys are masked out
        return self.ln2(x + self.drop(self.ffn(x)))


class TinyBERT(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 64,
        heads: int = 4,
        layers: int = 2,
        max_len: int = 32,
        dropout: float = 0.1,
        n_classes: int = 2,
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)  # learned absolute positions, as in BERT
        self.ln = nn.LayerNorm(d)
        self.layers = nn.ModuleList(EncoderLayer(d, heads, dropout) for _ in range(layers))
        self.classifier = nn.Sequential(
            nn.Linear(d, d), nn.Tanh(), nn.Linear(d, n_classes)
        )  # "pooler" on [CLS]
        self.mlm_head = nn.Linear(d, vocab)

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.ln(self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device)))
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Sentence logits from the [CLS] position (which can attend to every word)."""
        return self.classifier(self.encode(ids, mask)[:, 0])

    def mlm_logits(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.mlm_head(self.encode(ids, mask))


def mask_tokens(ids: torch.Tensor, vocab: int, p: float = 0.15, generator=None):
    """BERT masking: choose 15% of real tokens as targets; of those 80% -> [MASK],
    10% -> a random word, 10% unchanged. Labels are -100 (ignored) elsewhere."""
    real = ids >= len(SPECIALS)
    chosen = (torch.rand(ids.shape, generator=generator) < p) & real
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    roll = torch.rand(ids.shape, generator=generator)
    corrupted = ids.clone()
    corrupted[chosen & (roll < 0.8)] = MASK
    swap = chosen & (roll >= 0.8) & (roll < 0.9)
    corrupted[swap] = torch.randint(len(SPECIALS), vocab, (int(swap.sum()),), generator=generator)
    return corrupted, labels


def pretrain_mlm(
    model: TinyBERT, ids, mask, steps: int = 300, batch: int = 64, lr: float = 1e-3, seed: int = 0
) -> list[float]:
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history = []
    for _ in range(steps):
        idx = torch.randint(len(ids), (batch,), generator=gen)
        inp, labels = mask_tokens(ids[idx], model.mlm_head.out_features, generator=gen)
        logits = model.mlm_logits(inp, mask[idx])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    model.eval()
    return history


def fine_tune(
    model: TinyBERT,
    ids,
    mask,
    y,
    epochs: int = 15,
    batch: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(ids))
        total = 0.0
        for i in range(0, len(ids), batch):
            b = perm[i : i + batch]
            loss = F.cross_entropy(model(ids[b], mask[b]), y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        history.append(total / len(ids))
    model.eval()
    return history


@torch.no_grad()
def accuracy(model: TinyBERT, ids, mask, y) -> float:
    model.eval()
    return float((model(ids, mask).argmax(-1) == y).float().mean())
