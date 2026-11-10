import math
import random

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from s1_algorithms.dp_sequences.impl import lcs_length as reference_lcs

from .impl import (
    corpus_bleu,
    dcg,
    exact_match,
    lcs_length,
    mean_reciprocal_rank,
    ndcg,
    normalize_answer,
    recall_at_k,
    reciprocal_rank,
    rouge_l,
    token_f1,
)


def test_bleu_by_hand():
    hyp = "the cat sat on the mat".split()
    assert corpus_bleu([hyp], [[hyp]]) == pytest.approx(1.0)  # identical -> 1
    # Clipping: 7 x "the" against a reference with two "the" -> unigram precision 2/7.
    spam = ["the"] * 7
    refs = [["the", "cat", "is", "on", "the", "mat", "now"]]
    assert corpus_bleu([spam], [refs], max_n=1) == pytest.approx(2 / 7)
    # Brevity penalty: a correct but short hypothesis is penalised by exp(1 - r/c).
    short = "the cat".split()
    assert corpus_bleu([short], [[hyp]], max_n=1) == pytest.approx(math.exp(1 - 6 / 2))
    # No 4-gram overlap -> 0 unless smoothed.
    h, r = "a b c d e".split(), "a b c x e".split()
    assert corpus_bleu([h], [[r]]) == 0.0
    assert 0 < corpus_bleu([h], [[r]], smooth=True) < 1


def test_bleu_uses_best_reference_and_closest_length():
    hyp = "it is a guide to action".split()
    refs = [
        ["it", "is", "a", "guide", "to", "action", "that", "ensures"],
        ["it", "is", "a", "guide", "to", "action"],
    ]
    assert corpus_bleu([hyp], [refs]) == pytest.approx(1.0)  # closest reference has the same length


def test_rouge_l_and_lcs():
    rnd = random.Random(0)
    for _ in range(200):
        a = [rnd.choice("abcd") for _ in range(rnd.randint(0, 10))]
        b = [rnd.choice("abcd") for _ in range(rnd.randint(0, 10))]
        assert lcs_length(a, b) == reference_lcs(a, b)
    r = rouge_l("police killed the gunman".split(), "the police kill the gunman".split())
    assert r["precision"] == pytest.approx(3 / 4) and r["recall"] == pytest.approx(3 / 5)
    assert r["f"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert rouge_l(["x"], ["y"])["f"] == 0.0


def test_exact_match_and_f1():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("eiffel tower", "The Eiffel Tower.") == 1.0
    assert exact_match("Paris", ["London", "paris"]) == 1.0 and exact_match("Rome", "Paris") == 0.0
    assert token_f1("the big red barn", "red barn") == pytest.approx(2 * (2 / 3) * 1 / (2 / 3 + 1))
    assert token_f1("", "") == 1.0 and token_f1("cat", "dog") == 0.0


def test_ranking_metrics():
    ranked = ["d3", "d1", "d7", "d2"]
    assert (
        recall_at_k(ranked, {"d1", "d2"}, 2) == 0.5 and recall_at_k(ranked, {"d1", "d2"}, 4) == 1.0
    )
    assert reciprocal_rank(ranked, {"d7"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"zzz"}) == 0.0
    assert mean_reciprocal_rank([ranked, ranked], [{"d3"}, {"d1"}]) == pytest.approx(0.75)
    assert dcg([3, 2, 0]) == pytest.approx(3 + 2 / math.log2(3))


def test_ndcg_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(100):
        relevance = rng.integers(0, 4, size=10).astype(float)
        scores = rng.normal(size=10)
        order = np.argsort(-scores)
        for k in (3, 5, None):
            ours = ndcg(relevance[order].tolist(), relevance.tolist(), k)
            ref = ndcg_score(relevance[None], scores[None], k=k) if relevance.any() else 0.0
            assert ours == pytest.approx(ref)
    assert ndcg([3, 2, 1, 0]) == 1.0 and ndcg([0, 1, 2, 3]) < 1.0
