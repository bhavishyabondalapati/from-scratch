import math
import random
from pathlib import Path

import numpy as np
import pytest

from .impl import (
    BOS,
    EOS,
    NGramLM,
    build_vocab,
    generate,
    next_word_distribution,
    tune_interpolation,
)


def markov_corpus(n_sentences: int, seed: int, vocab_size: int = 30):
    """Sentences from a KNOWN bigram model, so the true per-token entropy is computable."""
    rng = np.random.default_rng(0)  # the true model is fixed; only the samples depend on `seed`
    words = [f"w{i}" for i in range(vocab_size)]
    states = [BOS, *words]
    outs = [*words, EOS]
    T = rng.dirichlet(np.full(len(outs), 0.3), size=len(states))  # sparse-ish transitions
    T[0, -1] = 0  # no empty sentences
    T[0] /= T[0].sum()
    sample_rng = np.random.default_rng(seed)
    sentences, logp, count = [], 0.0, 0
    for _ in range(n_sentences):
        s, state = [], 0
        while True:
            j = sample_rng.choice(len(outs), p=T[state])
            logp += math.log(T[state, j])
            count += 1
            if outs[j] == EOS or len(s) > 50:
                break
            s.append(outs[j])
            state = j + 1
        sentences.append(s)
    return sentences, math.exp(-logp / count)


@pytest.fixture(scope="module")
def corpus():
    train, _ = markov_corpus(3000, seed=1)
    test, true_ppl = markov_corpus(500, seed=2)
    return train, test, true_ppl


@pytest.mark.parametrize("smoothing", ["add_k", "interpolate", "kneser_ney"])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_distributions_sum_to_one(corpus, smoothing, n):
    train, _, _ = corpus
    model = NGramLM(n, smoothing).fit(train[:300])
    for history in [(BOS, BOS), ("w3", "w7"), ("w1", "never-seen-word")]:
        dist = next_word_distribution(model, history)
        assert sum(dist.values()) == pytest.approx(1.0)
        assert all(p > 0 for p in dist.values())  # smoothing: nothing has probability zero


def test_mle_assigns_zero_to_unseen_and_perplexity_is_infinite():
    train = [["a", "b"], ["a", "c"]]
    model = NGramLM(2, "mle").fit(train)
    assert model.prob("b", ("a",)) == 0.5
    assert model.perplexity([["b", "a"]]) == math.inf  # bigram "b a" never seen
    assert NGramLM(2, "add_k").fit(train).perplexity([["b", "a"]]) < math.inf


def test_bigram_kn_recovers_the_true_model(corpus):
    train, test, true_ppl = corpus
    kn = NGramLM(2, "kneser_ney").fit(train)
    assert kn.perplexity(test) == pytest.approx(true_ppl, rel=0.03)  # data really is bigram


def test_more_context_helps_and_kn_beats_add_one_on_sparse_data(corpus):
    train, test, true_ppl = corpus
    V = len(build_vocab(train))
    unigram = NGramLM(1, "add_k").fit(train).perplexity(test)
    bigram = {s: NGramLM(2, s).fit(train).perplexity(test) for s in ("add_k", "kneser_ney")}
    assert V > unigram > max(bigram.values())  # uniform > unigram > bigram
    assert min(bigram.values()) >= true_ppl * 0.99  # nothing beats the true model on fresh data
    # With little training data, smoothing matters more and Kneser-Ney wins.
    small = train[:300]
    assert NGramLM(2, "kneser_ney").fit(small).perplexity(test) < NGramLM(2, "add_k").fit(
        small
    ).perplexity(test)


def test_uniform_model_has_perplexity_v():
    model = NGramLM(1, "add_k", k=1e9).fit([["a", "b", "c"]])
    assert model.perplexity([["a", "c"]]) == pytest.approx(model.V, rel=1e-6)


def test_tuned_interpolation_beats_fixed_weights(corpus):
    train, test, _ = corpus
    lambdas = tune_interpolation(train[:2500], train[2500:], n=2)
    tuned = NGramLM(2, "interpolate", lambdas=lambdas).fit(train)
    fixed = NGramLM(2, "interpolate", lambdas=[0.2, 0.8]).fit(train)
    assert tuned.perplexity(test) < fixed.perplexity(test)


def test_generation_uses_only_vocabulary_words(corpus):
    train, *_ = corpus
    model = NGramLM(2, "kneser_ney").fit(train)
    words = generate(model, seed=0)
    assert all(w in model.vocab for w in words)


@pytest.mark.slow
def test_shakespeare_perplexities():
    from data_utils import tiny_shakespeare
    from s5_nlp.bow_tfidf.impl import tokenize

    lines = [tokenize(line) for line in tiny_shakespeare().splitlines() if line.strip()]
    lines = [line for line in lines if line]
    random.Random(0).shuffle(lines)
    train, test = lines[:25_000], lines[25_000:28_000]
    vocab = build_vocab(train, min_count=2)
    results = {
        f"{n}-gram {s}": NGramLM(n, s).fit(train, vocab).perplexity(test)
        for n in (1, 2, 3)
        for s in ("add_k", "kneser_ney")
    }
    assert results["2-gram kneser_ney"] < results["1-gram kneser_ney"]
    assert results["3-gram kneser_ney"] < 0.2 * results["3-gram add_k"]
    assert (
        results["2-gram add_k"] > results["1-gram add_k"]
    )  # add-1 over-smooths so badly it loses to unigrams
    lines_out = [f"{k}: {v:.1f}" for k, v in results.items()]
    (Path(__file__).parent / "shakespeare_perplexity.txt").write_text(
        f"vocab {len(vocab)} words (min count 2), train 25k lines, test 3k lines\n"
        + "\n".join(lines_out)
        + "\n"
    )
