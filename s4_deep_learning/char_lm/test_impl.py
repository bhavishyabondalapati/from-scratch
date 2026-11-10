from pathlib import Path

import numpy as np
import pytest

from s2_math.gradient_check.impl import numerical_gradient

from .impl import Adam, CharLSTM, Vocab, get_batch, train


def test_vocab_round_trip():
    v = Vocab("hello world")
    assert v.decode(v.encode("hello world")) == "hello world"
    assert len(v) == len(set("hello world"))


def test_batches_are_shifted_by_one():
    data = np.arange(100)
    x, y = get_batch(data, 10, 4, np.random.default_rng(0))
    assert x.shape == y.shape == (10, 4)
    np.testing.assert_array_equal(y, x + 1)


def test_whole_model_gradient_check(rng):
    model = CharLSTM(7, embed_dim=4, hidden=5, seed=1)
    x, y = rng.integers(0, 7, size=(6, 2)), rng.integers(0, 7, size=(6, 2))
    model.loss(x, y)
    model.backward()
    for params, grads, k in list(model.all_params()):
        analytic = grads[k].copy()
        numeric = numerical_gradient(lambda _: model.loss(x, y), params[k])
        # Norm-based relative error: some recurrent-weight gradients are ~1e-7, where an
        # element-wise ratio is dominated by floating-point noise.
        err = np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic + numeric)
        assert err < 1e-6, (k, err)


def test_memorises_a_short_repeating_text():
    text = "abcde fghij " * 40
    model, vocab, history, baseline = train(
        text, steps=300, seq_len=24, batch=8, hidden=32, lr=1e-2, eval_every=100
    )
    assert history["val"][-1] < 0.1  # a perfectly predictable sequence: loss -> 0
    assert history["val"][-1] < 0.05 * baseline
    assert model.sample(vocab, "abc", 9, temperature=0) == "de fghij "


def test_clipping_limits_update_size(rng):
    model = CharLSTM(5, embed_dim=3, hidden=4)
    for _, g, k in model.all_params():
        g[k] = rng.normal(size=g[k].shape) * 1e6  # a huge "exploding" gradient
    opt = Adam(model, lr=0.1)
    before = {(id(p), k): p[k].copy() for p, _, k in model.all_params()}
    norm = opt.step(clip=1.0)
    assert norm > 1e6
    for p, _, k in model.all_params():
        assert np.abs(p[k] - before[(id(p), k)]).max() <= 0.1 + 1e-9  # Adam caps each step at ~lr


@pytest.mark.slow
def test_tiny_shakespeare_beats_unigram_baseline():
    from data_utils import tiny_shakespeare

    text = tiny_shakespeare()[:300_000]
    model, vocab, history, baseline = train(text, steps=1500, seq_len=64, batch=32, hidden=128)
    final = history["val"][-1]
    assert final < 0.65 * baseline  # unigram ~3.3 nats/char
    sample = model.sample(vocab, "ROMEO:\n", 300, temperature=0.8)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history["step"], history["train"], label="train (smoothed)")
    ax.plot(history["step"], history["val"], label=f"val (final {final:.2f})")
    ax.axhline(baseline, ls="--", c="gray", label=f"unigram baseline {baseline:.2f}")
    ax.set(xlabel="step", ylabel="nats / char", title="NumPy char-LSTM on tiny-shakespeare")
    ax.legend()
    fig.tight_layout()
    here = Path(__file__).parent
    fig.savefig(here / "training_curve.png", dpi=110)
    plt.close(fig)
    (here / "sample.txt").write_text(
        f"val loss {final:.3f} nats/char, baseline {baseline:.3f}\n\n{sample}\n"
    )
