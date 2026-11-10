from pathlib import Path

import numpy as np
import pytest

from s2_math.gradient_check.impl import numerical_gradient
from s5_nlp.word2vec.impl import (
    Vocab,
    analogy,
    analogy_accuracy,
    planted_analogies,
    synthetic_corpus,
)

from .impl import GloVe, cooccurrence_matrix, ppmi_svd_embeddings, weighting


def test_cooccurrence_by_hand():
    v = Vocab([["a", "b", "c"]])
    X = cooccurrence_matrix([["a", "b", "c"]], v, window=2)
    # a-b at distance 1, b-c at distance 1, a-c at distance 2.
    np.testing.assert_allclose(X, [[0, 1, 0.5], [1, 0, 1], [0.5, 1, 0]])


def test_weighting_function():
    X = np.array([0.0, 1.0, 50.0, 100.0, 1000.0])
    np.testing.assert_allclose(weighting(X), [0, 0.01**0.75, 0.5**0.75, 1, 1])


def test_gradients_match_finite_differences(rng):
    X = rng.poisson(2.0, size=(6, 6)).astype(float)
    X = X + X.T
    model = GloVe(6, dim=3, seed=1)
    model.b, model.bc = rng.normal(size=6), rng.normal(size=6)
    F = weighting(X, x_max=5)
    _, grads = model.loss_and_grads(X, F)
    for name in ("W", "C", "b", "bc"):
        param = getattr(model, name)
        numeric = numerical_gradient(lambda _: model.loss_and_grads(X, F)[0], param)
        assert np.linalg.norm(grads[name] - numeric) / np.linalg.norm(grads[name] + numeric) < 1e-7


@pytest.fixture(scope="module")
def trained():
    corpus = synthetic_corpus(4000, seed=0)
    vocab = Vocab(corpus)
    X = cooccurrence_matrix(corpus, vocab, window=5)
    model = GloVe(len(vocab), dim=30, seed=0)
    history = model.fit(X, epochs=400, x_max=10)
    return model, vocab, X, history


def test_fit_reduces_loss_and_fits_log_counts(trained):
    model, _, X, history = trained
    assert history[-1] < 0.05 * history[0]
    pred = model.W @ model.C.T + model.b[:, None] + model.bc[None, :]
    big = X > 20  # well-observed pairs should be fitted closely
    assert np.corrcoef(pred[big], np.log(X[big]))[0, 1] > 0.95


def test_glove_analogies_across_seeds_vs_count_baseline(trained):
    model, vocab, X, history = trained
    qs = planted_analogies()
    accs = []
    for seed in range(5):
        m = GloVe(len(vocab), dim=30, seed=seed)
        m.fit(X, epochs=400, x_max=10)
        accs.append(analogy_accuracy(m.embeddings, vocab, qs))
    ppmi_acc = analogy_accuracy(ppmi_svd_embeddings(X, 30), vocab, qs)
    rand_acc = analogy_accuracy(np.random.default_rng(0).normal(size=(len(vocab), 30)), vocab, qs)
    assert analogy(model.embeddings, vocab, "man", "king", "woman") == ["queen"]
    assert np.mean(accs) > 0.75 and min(accs) > 0.5 > rand_acc  # far above chance (~4%)...
    assert ppmi_acc == 1.0  # ...but on this tiny corpus the count baseline is more reliable

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(history)
    ax.set(
        xlabel="AdaGrad step (full batch)",
        ylabel="weighted squared error",
        title=f"GloVe analogies {np.mean(accs):.0%} avg of 5 seeds (PPMI-SVD {ppmi_acc:.0%})",
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print("glove", [round(a, 2) for a in accs], "ppmi-svd", ppmi_acc, "random", rand_acc)
