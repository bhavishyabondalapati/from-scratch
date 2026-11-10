"""Okapi BM25 ranking from scratch, with an inverted index."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

STOPWORDS = set(
    "a an the and or of to in on for is are was were be by with as at it this that from what why how "
    "when which do does you your can its into than then there their them they we our not no if".split()
)


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if not (remove_stopwords and t in STOPWORDS)]


class BM25:
    """score(q, d) = sum over query terms t of

        idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    - tf saturates: the 10th occurrence of a word adds far less than the 1st (k1 controls how fast).
    - length normalisation: a long document is expected to contain more matches (b controls how much).
    - idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)): rare terms matter more (Lucene's always-positive form).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b

    def fit(self, docs: list[list[str]]) -> BM25:
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avgdl = sum(self.doc_len) / max(self.N, 1)
        # Inverted index: term -> {doc id: term frequency}. Scoring only touches docs
        # that contain at least one query term.
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term][i] = tf
        self.idf = {
            t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }
        return self

    def term_score(self, tf: int, dl: int, idf: float) -> float:
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        return idf * tf * (self.k1 + 1) / (tf + norm)

    def scores(self, query: list[str]) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in query:  # a repeated query term counts again, as in the classic formula
            for doc, tf in self.postings.get(term, {}).items():
                out[doc] += self.term_score(tf, self.doc_len[doc], self.idf[term])
        return out

    def search(self, query: list[str], k: int = 10) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda x: (-x[1], x[0]))
        return ranked[:k]


def brute_force_score(
    query: list[str], doc: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75
) -> float:
    """The formula written out directly, for testing the indexed version."""
    N, avgdl = len(docs), sum(map(len, docs)) / len(docs)
    total = 0.0
    for t in query:
        df = sum(t in d for d in docs)
        if df == 0:
            continue
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        tf = doc.count(t)
        total += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / avgdl))
    return total
