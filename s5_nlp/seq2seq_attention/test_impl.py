from pathlib import Path

import pytest
import torch

from .impl import (
    EOS,
    PAD,
    AdditiveAttention,
    Seq2Seq,
    decode,
    encode,
    exact_match,
    make_batch,
    train,
)


def test_data_pipeline():
    assert decode(encode("abcj") + [EOS, 5]) == "abcj"
    src, lengths, tgt = make_batch(5, 3, 6, torch.Generator().manual_seed(0))
    for s, L, t in zip(src, lengths, tgt):
        assert decode(t.tolist()) == decode(s[:L].tolist())[::-1]
        assert (s[L:] == PAD).all() and t[L] == EOS


def test_additive_attention_is_a_masked_distribution():
    torch.manual_seed(0)
    attn = AdditiveAttention(4, 6, 5)
    H = torch.randn(2, 7, 6)
    mask = torch.ones(2, 7, dtype=torch.bool)
    mask[1, 4:] = False
    ctx, w = attn(torch.randn(2, 4), H, mask)
    torch.testing.assert_close(w.sum(-1), torch.ones(2))
    assert torch.all(w[1, 4:] == 0)
    torch.testing.assert_close(ctx, (w[:, :, None] * H).sum(1))


def test_additive_attention_gradcheck():
    torch.manual_seed(1)
    attn = AdditiveAttention(3, 4, 5).double()
    s = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    H = torch.randn(2, 6, 4, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)
    assert torch.autograd.gradcheck(lambda a, b: attn(a, b, mask)[0], (s, H))


def test_shapes_and_padding_does_not_change_the_encoding():
    torch.manual_seed(2)
    model = Seq2Seq()
    src, lengths, tgt = make_batch(4, 3, 8, torch.Generator().manual_seed(1))
    assert model(src, lengths, tgt).shape == (4, tgt.shape[1], model.out.out_features)
    # Extra padding at the end must not affect a sequence's encoder output (packing ignores it).
    padded = torch.cat([src, torch.full((4, 5), PAD)], dim=1)
    H1, s1, _ = model.encode(src, lengths)
    H2, s2, _ = model.encode(padded, lengths)
    torch.testing.assert_close(s1, s2)
    torch.testing.assert_close(H1, H2[:, : src.shape[1]])


@pytest.mark.slow
def test_attention_reverses_long_strings_far_better():
    with_attn, without = Seq2Seq(attention=True), Seq2Seq(attention=False)
    hist_a = train(with_attn, steps=2500)
    hist_n = train(without, steps=2500)
    lengths = [4, 6, 8, 10, 12]
    acc_a = [exact_match(with_attn, 300, L) for L in lengths]
    acc_n = [exact_match(without, 300, L) for L in lengths]
    assert acc_a[-1] > 0.85  # 12 letters reversed exactly; chance is 10^-12
    assert acc_a[-1] > acc_n[-1] + 0.3
    # The learned alignment is the anti-diagonal: output step t looks at input position L-1-t.
    src, lengths_t, _ = make_batch(1, 10, 10, torch.Generator().manual_seed(9))
    _, attn = with_attn.greedy_decode(src, lengths_t, 10)
    peaks = attn[0].argmax(-1)
    assert (peaks == torch.arange(9, -1, -1)).float().mean() > 0.8

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    smooth = lambda h: torch.tensor(h).unfold(0, 50, 50).mean(1)  # noqa: E731
    axes[0].plot(smooth(hist_a), label="with attention")
    axes[0].plot(smooth(hist_n), label="no attention")
    axes[0].set(xlabel="step / 50", ylabel="loss", yscale="log", title="training")
    axes[0].legend()
    axes[1].plot(lengths, acc_a, marker="o", label="with attention")
    axes[1].plot(lengths, acc_n, marker="o", label="no attention")
    axes[1].set(
        xlabel="string length", ylabel="exact match", title="reversal accuracy", ylim=(0, 1.05)
    )
    axes[1].legend()
    axes[2].imshow(attn[0], cmap="viridis")
    axes[2].set(xlabel="input position", ylabel="output step", title="attention weights")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print("attn", acc_a, "no-attn", acc_n)
