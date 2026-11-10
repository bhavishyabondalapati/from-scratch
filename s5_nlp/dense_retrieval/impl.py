"""Dense retrieval: a small trained embedding model, plus brute-force and IVF vector indexes."""

from __future__ import annotations

import random
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from s3_classical_ml.kmeans.impl import KMeans
from s5_nlp.bm25.impl import tokenize

# --- A small embedding model (dual encoder) --------------------------------------------


class TextEncoder(nn.Module):
    """text -> unit vector. Mean of learned word embeddings, then a linear layer.

    Small on purpose (no download needed); the same model encodes queries and passages
    (a "bi-encoder"), so passage vectors can be computed once and indexed.
    """

    def __init__(self, vocab: dict[str, int], dim: int = 64) -> None:
        super().__init__()
        self.vocab = vocab
        self.emb = nn.EmbeddingBag(len(vocab) + 1, dim, mode="mean", padding_idx=0)  # 0 = unknown
        self.proj = nn.Linear(dim, dim)

    def ids(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        flat, offsets = [], []
        for t in texts:
            offsets.append(len(flat))
            toks = [self.vocab.get(w, 0) for w in tokenize(t)] or [0]
            flat.extend(toks)
        return torch.tensor(flat), torch.tensor(offsets)

    def forward(self, texts: list[str]) -> torch.Tensor:
        return F.normalize(self.proj(self.emb(*self.ids(texts))), dim=-1)

    @torch.no_grad()
    def encode(self, texts: list[str], batch: int = 256) -> np.ndarray:
        self.eval()
        return np.concatenate(
            [self(texts[i : i + batch]).numpy() for i in range(0, len(texts), batch)]
        )


def info_nce(q: torch.Tensor, p: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
    """Contrastive loss with in-batch negatives: query i should match passage i and
    not the other B-1 passages in the batch. Cross-entropy over the similarity matrix."""
    logits = q @ p.T / temperature
    return F.cross_entropy(logits, torch.arange(len(q)))


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 4]


def train_encoder(
    passages, steps: int = 1500, batch: int = 64, dim: int = 128, lr: float = 1e-3, seed: int = 0
):
    """Self-supervised pairs from the corpus itself: (a sentence of a passage, a DIFFERENT
    passage of the same topic). No queries or labels are used."""
    rnd = random.Random(seed)
    torch.manual_seed(seed)
    words = sorted({w for p in passages for w in tokenize(p.title + " " + p.text)})
    model = TextEncoder({w: i + 1 for i, w in enumerate(words)}, dim)
    by_topic: dict[str, list] = {}
    for p in passages:
        by_topic.setdefault(p.topic, []).append(p)
    topics = [t for t, ps in by_topic.items() if len(ps) > 1]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    model.train()
    for _ in range(steps):
        chosen = rnd.sample(
            topics, min(batch, len(topics))
        )  # one pair per topic: no false negatives
        anchors, positives = [], []
        for t in chosen:
            a, b = rnd.sample(by_topic[t], 2)
            anchors.append(rnd.choice(sentences(a.text) or [a.text]))
            positives.append(b.title + " " + b.text)
        loss = info_nce(model(anchors), model(positives))
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append(loss.item())
    return model.eval(), history


# --- Vector indexes ----------------------------------------------------------------------------


class BruteForceIndex:
    """Exact search: compare the query with every stored vector (inner product = cosine
    for unit vectors). O(N d) per query, perfect recall."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        sims = np.asarray(queries, dtype=np.float32) @ self.vectors.T
        k = min(k, len(self.vectors))
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]  # unordered top k in O(N)
        order = np.argsort(-np.take_along_axis(sims, top, 1), axis=1)
        idx = np.take_along_axis(top, order, 1)
        return idx, np.take_along_axis(sims, idx, 1)


class IVFIndex:
    """Inverted file index: cluster the vectors with k-means into `n_lists` cells. A query
    is compared only with the vectors in its `n_probe` nearest cells.
    Approximate: a true neighbour sitting in an unprobed cell is missed.
    Cost per query ~ n_lists + (n_probe / n_lists) * N comparisons instead of N."""

    def __init__(self, vectors: np.ndarray, n_lists: int = 32, seed: int = 0) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        km = KMeans(n_lists, n_init=1, max_iter=25, random_state=seed).fit(self.vectors)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self.lists = [np.flatnonzero(km.labels_ == c) for c in range(n_lists)]
        self.last_n_compared = 0

    def search(
        self, queries: np.ndarray, k: int, n_probe: int = 4
    ) -> tuple[np.ndarray, np.ndarray]:
        queries = np.asarray(queries, dtype=np.float32)
        # Nearest cells by distance to the centroid (centroids are not unit length).
        d2 = (
            (queries**2).sum(1)[:, None]
            - 2 * queries @ self.centroids.T
            + (self.centroids**2).sum(1)[None, :]
        )
        probes = np.argsort(d2, axis=1)[:, :n_probe]
        all_idx = np.full((len(queries), k), -1)
        all_sims = np.full((len(queries), k), -np.inf, dtype=np.float32)
        compared = 0
        for qi, cells in enumerate(probes):
            cand = np.concatenate([self.lists[c] for c in cells])
            compared += len(cand)
            sims = self.vectors[cand] @ queries[qi]
            top = np.argsort(-sims)[:k]
            all_idx[qi, : len(top)] = cand[top]
            all_sims[qi, : len(top)] = sims[top]
        self.last_n_compared = compared / len(queries)
        return all_idx, all_sims


def recall_vs_exact(approx: np.ndarray, exact: np.ndarray) -> float:
    """Fraction of the exact top-k neighbours that the approximate search also returned."""
    return float(np.mean([len(set(a) & set(e)) / len(e) for a, e in zip(approx, exact)]))
