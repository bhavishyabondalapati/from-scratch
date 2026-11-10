import math
import time
from pathlib import Path

import pytest
import torch

from .impl import (
    GPT,
    CharTokenizer,
    GPTConfig,
    filter_top_k,
    filter_top_p,
    generate,
    get_batch,
    sample_next,
    train,
)

TINY = GPTConfig(vocab_size=20, block_size=32, n_layer=2, n_head=2, d_model=32, dropout=0.0)


def test_shapes_loss_at_init_and_weight_tying():
    torch.manual_seed(0)
    model = GPT(TINY)
    idx = torch.randint(0, 20, (3, 10))
    targets = torch.randint(0, 20, (3, 10))  # independent of the inputs
    logits, loss = model(idx, targets)
    assert logits.shape == (3, 10, 20)
    assert loss.item() == pytest.approx(math.log(20), rel=0.05)  # untrained ~ uniform guess
    assert model.head.weight is model.tok_emb.weight


def test_causality_future_tokens_do_not_change_the_past():
    torch.manual_seed(1)
    model = GPT(TINY).eval()
    a = torch.randint(0, 20, (1, 12))
    b = a.clone()
    b[0, 8:] = torch.randint(0, 20, (4,))  # change only the last 4 tokens
    la, _ = model(a)
    lb, _ = model(b)
    torch.testing.assert_close(la[0, :8], lb[0, :8])


def test_kv_cache_matches_full_recompute():
    torch.manual_seed(2)
    model = GPT(TINY).eval()
    prompt = torch.randint(0, 20, (2, 5))
    with_cache = generate(model, prompt, 20, use_cache=True, temperature=0)
    without = generate(model, prompt, 20, use_cache=False, temperature=0)
    torch.testing.assert_close(with_cache, without)
    # Incremental logits equal the full forward pass at every position.
    seq = with_cache[:1]
    full, _ = model(seq)
    caches = [{} for _ in model.blocks]
    steps = [model(seq[:, :5], caches=caches)[0]]
    for t in range(5, seq.shape[1]):
        steps.append(model(seq[:, t : t + 1], caches=caches)[0])
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-5)


def test_kv_cache_is_faster():
    cfg = GPTConfig(vocab_size=20, block_size=256, n_layer=4, n_head=4, d_model=128, dropout=0.0)
    model = GPT(cfg).eval()
    prompt = torch.zeros(1, 1, dtype=torch.long)
    timings = {}
    for use_cache in (True, False):
        t = time.perf_counter()
        generate(model, prompt, 200, use_cache=use_cache, temperature=0)
        timings[use_cache] = time.perf_counter() - t
    assert timings[True] < timings[False]


def test_sampling_filters():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0, 0.0]])
    kept = filter_top_k(logits, 2)
    assert torch.isfinite(kept).sum() == 2 and torch.isfinite(kept[0, :2]).all()
    probs = torch.softmax(logits, -1)[0]  # [0.53, 0.20, 0.12, 0.03, 0.07]
    nucleus = filter_top_p(logits, 0.7)
    assert torch.isfinite(nucleus[0]).tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]  # 0.53 + 0.20 >= 0.7
    assert torch.isfinite(filter_top_p(logits, 0.01)).sum() == 1  # top token always survives
    assert probs.argmax() == sample_next(logits, temperature=0)
    gen = torch.Generator().manual_seed(0)
    draws = torch.stack([sample_next(logits, top_k=2, generator=gen) for _ in range(500)])
    assert set(draws.flatten().tolist()) == {0, 1}
    cold = torch.stack([sample_next(logits, temperature=0.05, generator=gen) for _ in range(200)])
    assert (cold == 0).float().mean() > 0.99  # low temperature ~ greedy


def test_overfits_a_tiny_repeating_text():
    text = "to be or not to be, that is the question. " * 20
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text))
    cfg = GPTConfig(
        vocab_size=len(tok.chars), block_size=32, n_layer=2, n_head=2, d_model=64, dropout=0.0
    )
    model = GPT(cfg)
    hist = train(model, data, data, steps=300, batch=16, lr=3e-3, eval_every=100)
    assert hist["val"][-1] < 0.1
    prompt = torch.tensor([tok.encode("to be or ")])
    out = tok.decode(generate(model, prompt, 15, temperature=0)[0].tolist())
    assert out.startswith("to be or not to be, that")


def test_get_batch_shift():
    data = torch.arange(100)
    x, y = get_batch(data, 8, 4, torch.Generator().manual_seed(0))
    torch.testing.assert_close(y, x + 1)


@pytest.mark.slow
def test_shakespeare_gpt():
    from .impl import pretrained_shakespeare_gpt

    model, tok, history = pretrained_shakespeare_gpt()
    final_val = history["val"][-1]
    assert final_val < 1.6  # unigram baseline 3.3; the NumPy char-LSTM got 1.63 on less data
    prompt = torch.tensor([tok.encode("ROMEO:\n")])
    sample = tok.decode(generate(model, prompt, 120, temperature=0.8, top_k=20, seed=0)[0].tolist())

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(history["step"], history["train"], label="train")
    ax.plot(history["step"], history["val"], label=f"val (final {final_val:.3f})")
    ax.set(
        xlabel="step",
        ylabel="nats / char",
        title=f"char GPT ({model.n_params() / 1e6:.2f}M params)",
    )
    ax.legend()
    fig.tight_layout()
    here = Path(__file__).parent
    fig.savefig(here / "training_curve.png", dpi=110)
    plt.close(fig)
    (here / "sample.txt").write_text(f"val loss {final_val:.3f} nats/char\n\n{sample}\n")
