"""A minimal retrieval-augmented generation (RAG) loop with an OFFLINE extractive answerer.

retrieve (BM25 over the repo's notes) -> pick the best sentence from the retrieved
passages -> return it with a citation. No language model or API is involved: the
"generator" is extractive, so every answer is a quoted sentence with a source.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from s5_nlp.bm25.impl import BM25, tokenize
from s5_nlp.corpus import Passage


@dataclass
class Answer:
    text: str
    source: str  # passage id, for citation
    score: float
    retrieved: list[str]  # passage ids that were considered


def split_sentences(text: str) -> list[str]:
    # Split on sentence punctuation and on table-row / bullet boundaries left in the notes.
    parts = re.split(r"(?<=[.!?])\s+|\s+(?=- )|\s*\|\s*", text)
    return [p.strip(" -") for p in parts if len(p.split()) >= 5]


class RAG:
    def __init__(self, passages: list[Passage], k: int = 5, passage_weight: float = 1.0) -> None:
        self.passages, self.k, self.passage_weight = passages, k, passage_weight
        docs = [tokenize(p.title + " " + p.text) for p in passages]
        self.retriever = BM25().fit(docs)
        # Document frequencies for weighting overlap in the sentence scorer.
        self.N = len(docs)
        self.df = Counter(t for d in docs for t in set(d))

    def idf(self, term: str) -> float:
        return math.log(1 + self.N / (1 + self.df.get(term, 0)))

    def retrieve(self, question: str) -> list[tuple[Passage, float]]:
        return [(self.passages[i], s) for i, s in self.retriever.search(tokenize(question), self.k)]

    def sentence_score(
        self, question_terms: set[str], sentence: str, passage_score: float, best_passage: float
    ) -> float:
        """idf-weighted fraction of question terms present in the sentence, plus a bonus for
        coming from a highly ranked passage (context matters as well as word overlap)."""
        words = set(tokenize(sentence))
        if not question_terms:
            return 0.0
        overlap = sum(self.idf(t) for t in question_terms & words) / sum(
            self.idf(t) for t in question_terms
        )
        return overlap + self.passage_weight * passage_score / max(best_passage, 1e-9)

    def answer(self, question: str) -> Answer:
        hits = self.retrieve(question)
        if not hits:
            return Answer("I could not find this in the notes.", "", 0.0, [])
        q_terms = set(tokenize(question))
        best_passage = hits[0][1]
        best = max(
            (
                (s, p.pid, self.sentence_score(q_terms, s, ps, best_passage))
                for p, ps in hits
                for s in split_sentences(p.text)
            ),
            key=lambda x: x[2],
            default=(hits[0][0].text, hits[0][0].pid, 0.0),
        )
        return Answer(best[0], best[1], best[2], [p.pid for p, _ in hits])

    def format(self, question: str) -> str:
        a = self.answer(question)
        return f"Q: {question}\nA: {a.text}\n[source: {a.source}]"
