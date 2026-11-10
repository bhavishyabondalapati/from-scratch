from pathlib import Path

import numpy as np
import pytest

from s2_math.gradient_check.impl import numerical_gradient

from .impl import (
    SkipGramNS,
    Vocab,
    analogy,
    analogy_accuracy,
    most_similar,
    noise_distribution,
    planted_analogies,
    skipgram_pairs,
    synthetic_corpus,
    train_word2vec,
)


def test_pairs_and_noise_distribution():
    rng = np.random.default_rng(0)
    pairs = skipgram_pairs([[0, 1, 2, 3]], window=1, rng=rng)
    assert {tuple(p) for p in pairs} == {(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)}
    p = noise_distribution(np.array([1.0, 16.0]))
    assert p.sum() == pytest.approx(1)
    assert p[1] / p[0] == pytest.approx(16**0.75)  # 8x, not 16x: rare words are boosted


def test_sgns_gradients_match_finite_differences(rng):
    model = SkipGramNS(10, dim=4, seed=0)
    model.W_out = rng.normal(size=model.W_out.shape)
    centers, contexts = np.array([1, 2, 1]), np.array([3, 4, 5])
    negatives = np.array([[6, 7], [8, 9], [6, 0]])
    _, dv, du_pos, du_neg = model.loss_and_grads(centers, contexts, negatives)
    # Scatter the per-example gradients into full-table gradients, then compare.
    g_in, g_out = np.zeros_like(model.W_in), np.zeros_like(model.W_out)
    np.add.at(g_in, centers, dv)
    np.add.at(g_out, contexts, du_pos)
    np.add.at(g_out, negatives, du_neg)

    def loss(_):
        return model.loss_and_grads(centers, contexts, negatives)[0]

    for table, analytic in [(model.W_in, g_in), (model.W_out, g_out)]:
        numeric = numerical_gradient(loss, table)
        assert np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic + numeric) < 1e-7


def test_planted_analogy_set():
    qs = planted_analogies()
    assert ("man", "king", "woman", "queen") in qs and ("king", "prince", "queen", "princess") in qs
    assert len(qs) == len(set(qs))


@pytest.fixture(scope="module")
def trained():
    corpus = synthetic_corpus(4000, seed=0)
    return train_word2vec(corpus, dim=30, epochs=5, seed=0)


def test_learns_planted_structure(trained):
    model, vocab, history = trained
    E = model.W_in
    assert history[-1] < history[0]
    assert analogy(E, vocab, "man", "king", "woman") == ["queen"]
    acc = analogy_accuracy(E, vocab, planted_analogies())
    random_acc = analogy_accuracy(
        np.random.default_rng(0).normal(size=E.shape), vocab, planted_analogies()
    )
    assert acc > 0.9 and random_acc < 0.3
    # Nearest neighbours share attributes: "he" is closest to other male context words.
    neighbours = {w for w, _ in most_similar(E, vocab, "he", 3)}
    assert neighbours <= {"his", "him", "himself", "king", "prince", "man", "boy"}

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(history) + 1), history, marker="o")
    ax.set(
        xlabel="epoch", ylabel="SGNS loss", title=f"word2vec, planted analogies {acc:.0%} correct"
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)


def test_vocab():
    v = Vocab([["a", "b", "a"], ["c"]], min_count=1)
    assert v.words == ["a", "b", "c"] and list(v.counts) == [2, 1, 1]
    assert v.encode(["a", "zzz", "c"]) == [0, 2]
