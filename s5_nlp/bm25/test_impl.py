import random

import numpy as np
import pytest

from s5_nlp.bow_tfidf.impl import TfidfVectorizer
from s5_nlp.corpus import load_corpus, relevant_ids
from s5_nlp.eval_metrics.impl import mean_reciprocal_rank, recall_at_k

from .impl import BM25, brute_force_score, tokenize


def test_indexed_scores_match_the_formula():
    rnd = random.Random(0)
    vocab = list("abcdefgh")
    docs = [[rnd.choice(vocab) for _ in range(rnd.randint(1, 15))] for _ in range(30)]
    model = BM25().fit(docs)
    for _ in range(50):
        q = [rnd.choice(vocab + ["zzz"]) for _ in range(rnd.randint(1, 4))]
        scores = model.scores(q)
        for i, d in enumerate(docs):
            assert scores.get(i, 0.0) == pytest.approx(brute_force_score(q, d, docs))


def test_term_frequency_saturates_and_length_normalises():
    model = BM25(k1=1.2, b=0.75).fit([["x"] * 10, ["y"]])
    s = [model.term_score(tf, 10, 1.0) for tf in (1, 2, 5, 50, 5000)]
    assert all(a < b for a, b in zip(s, s[1:]))  # more matches is better...
    assert s[-1] < 1.2 + 1  # ...but bounded by k1 + 1 (saturation)
    t1, t2, t3 = (model.term_score(tf, 10, 1.0) for tf in (1, 2, 3))
    assert t3 - t2 < t2 - t1  # diminishing returns per extra occurrence
    short, long_ = model.term_score(2, 5, 1.0), model.term_score(2, 50, 1.0)
    assert short > long_  # same matches in a longer document count less
    no_norm = BM25(b=0.0).fit([["x"] * 10, ["y"]])
    assert no_norm.term_score(2, 5, 1.0) == no_norm.term_score(2, 50, 1.0)


def test_rare_terms_weigh_more():
    docs = [["common", "rare"]] + [["common", "filler"]] * 20
    model = BM25().fit(docs)
    assert model.idf["rare"] > model.idf["common"] > 0
    assert model.search(["common", "rare"], k=1)[0][0] == 0


def test_retrieval_on_the_repository_notes():
    passages, queries = load_corpus()
    model = BM25().fit([tokenize(p.title + " " + p.text) for p in passages])
    rankings, relevants = [], []
    for q in queries:
        hits = model.search(tokenize(q.text), k=10)
        rankings.append([passages[i].pid for i, _ in hits])
        relevants.append(relevant_ids(passages, q))
    mrr = mean_reciprocal_rank(rankings, relevants)
    r10 = np.mean([recall_at_k(r, rel, 10) > 0 for r, rel in zip(rankings, relevants)])

    # Baseline: TF-IDF cosine similarity.
    vec = TfidfVectorizer().fit([p.title + " " + p.text for p in passages])
    P = vec.transform([p.title + " " + p.text for p in passages])
    Q = vec.transform([q.text for q in queries])
    tfidf_rankings = [[passages[i].pid for i in np.argsort(-row)[:10]] for row in Q @ P.T]
    tfidf_mrr = mean_reciprocal_rank(tfidf_rankings, relevants)
    n_topics = len({p.topic for p in passages})
    assert mrr > 0.6 and r10 > 0.8  # random ranking would give MRR around 1/n_topics
    assert mrr > tfidf_mrr
    assert 1 / n_topics < 0.05
    from pathlib import Path

    (Path(__file__).parent / "results.txt").write_text(
        f"passages: {len(passages)} from {n_topics} topics, held-out questions: {len(queries)}\n"
        f"MRR@10 BM25 {mrr:.3f}, TF-IDF {tfidf_mrr:.3f}; relevant passage in top 10: {r10:.3f}\n"
    )
