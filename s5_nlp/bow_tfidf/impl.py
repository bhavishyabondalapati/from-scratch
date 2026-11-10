"""Bag-of-words counts and TF-IDF, matching scikit-learn's default conventions."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # sklearn's default: words of 2+ word characters


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n_range: tuple[int, int]) -> list[str]:
    lo, hi = n_range
    return [
        " ".join(tokens[i : i + n]) for n in range(lo, hi + 1) for i in range(len(tokens) - n + 1)
    ]


class CountVectorizer:
    """Each document becomes a vector of term counts over a fixed vocabulary.
    Word ORDER is thrown away: "dog bites man" == "man bites dog"."""

    def __init__(
        self, ngram_range: tuple[int, int] = (1, 1), min_df: int = 1, binary: bool = False
    ) -> None:
        self.ngram_range, self.min_df, self.binary = ngram_range, min_df, binary

    def _terms(self, doc: str) -> list[str]:
        return ngrams(tokenize(doc), self.ngram_range)

    def fit(self, docs: list[str]) -> CountVectorizer:
        df = Counter(t for d in docs for t in set(self._terms(d)))
        terms = sorted(t for t, c in df.items() if c >= self.min_df)  # sorted, like sklearn
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        X = np.zeros((len(docs), len(self.vocabulary_)))
        for row, d in enumerate(docs):
            for t, c in Counter(self._terms(d)).items():
                j = self.vocabulary_.get(t)
                if j is not None:  # words never seen in training are ignored
                    X[row, j] = 1 if self.binary else c
        return X

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


class TfidfVectorizer:
    """tf-idf(t, d) = tf(t, d) * idf(t), then each row scaled to unit L2 norm.

    idf(t) = ln((1 + n) / (1 + df(t))) + 1   (smooth_idf: as if one extra document
    contained every term, so no division by zero; the +1 keeps terms that are in
    every document from being zeroed out completely).
    Rare terms get large weights; terms in almost every document ("the") get small ones.
    sublinear_tf replaces tf by 1 + ln(tf), so the 10th repeat counts less than the 1st.
    """

    def __init__(
        self,
        ngram_range=(1, 1),
        min_df: int = 1,
        sublinear_tf: bool = False,
        smooth_idf: bool = True,
    ) -> None:
        self.counts = CountVectorizer(ngram_range, min_df)
        self.sublinear_tf, self.smooth_idf = sublinear_tf, smooth_idf

    def fit(self, docs: list[str]) -> TfidfVectorizer:
        X = self.counts.fit_transform(docs)
        n, df = len(docs), (X > 0).sum(axis=0)
        s = 1 if self.smooth_idf else 0
        self.idf_ = np.log((n + s) / (df + s)) + 1
        self.vocabulary_ = self.counts.vocabulary_
        return self

    def transform(self, docs: list[str]) -> np.ndarray:
        tf = self.counts.transform(docs)
        if self.sublinear_tf:
            tf = np.where(tf > 0, 1 + np.log(np.where(tf > 0, tf, 1)), 0)
        X = tf * self.idf_
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1, norms)

    def fit_transform(self, docs: list[str]) -> np.ndarray:
        return self.fit(docs).transform(docs)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def top_terms(vectorizer: TfidfVectorizer, row: np.ndarray, k: int = 5) -> list[str]:
    """The k highest-weighted terms of one TF-IDF vector: a cheap document summary."""
    inv = {i: t for t, i in vectorizer.vocabulary_.items()}
    return [inv[i] for i in np.argsort(-row)[:k] if row[i] > 0]


def idf_by_hand(term: str, docs: list[str]) -> float:
    """For the tests: the smoothed idf formula written out for a single term."""
    n = len(docs)
    df = sum(term in set(tokenize(d)) for d in docs)
    return math.log((1 + n) / (1 + df)) + 1
