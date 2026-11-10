import math

import pytest
import torch
import torch.nn.functional as F

from .impl import MultiHeadAttention, scaled_dot_product_attention


@pytest.mark.parametrize("causal", [False, True])
def test_matches_pytorch_sdpa(causal):
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 8), torch.randn(2, 3, 5, 6)
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, causal=causal),
        F.scaled_dot_product_attention(q, k, v, is_causal=causal),
    )


def test_boolean_mask_matches_pytorch():
    torch.manual_seed(1)
    q, k, v = torch.randn(2, 4, 8), torch.randn(2, 6, 8), torch.randn(2, 6, 8)
    mask = torch.rand(2, 4, 6) > 0.3
    mask[..., 0] = True  # keep at least one key per row
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
    )


def test_weights_are_a_distribution_and_respect_masks():
    torch.manual_seed(2)
    q = k = v = torch.randn(1, 6, 4)
    _, w = scaled_dot_product_attention(q, k, v, causal=True, return_weights=True)
    torch.testing.assert_close(w.sum(-1), torch.ones(1, 6))
    assert torch.all(w.triu(1) == 0)  # nothing attends to the future
    torch.testing.assert_close(
        w[0, 0], torch.tensor([1.0, 0, 0, 0, 0, 0])
    )  # first token sees only itself
    # Fully masked rows give zeros rather than NaN.
    out = scaled_dot_product_attention(q, k, v, mask=torch.zeros(1, 6, 6, dtype=torch.bool))
    assert torch.all(out == 0)


def test_scaling_keeps_softmax_from_saturating():
    torch.manual_seed(3)
    d = 512
    q, k = torch.randn(1, 10, d), torch.randn(1, 10, d)
    raw = torch.softmax(q @ k.transpose(-2, -1), dim=-1)
    scaled = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d), dim=-1)
    assert raw.max(-1).values.mean() > 0.8  # unscaled: scores have std ~sqrt(d), mostly one-hot
    assert scaled.max(-1).values.mean() < 0.5  # scaled: scores have std ~1


def test_gradcheck():
    torch.manual_seed(4)
    q, k, v = (torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True) for _ in range(3))
    assert torch.autograd.gradcheck(
        lambda a, b, c: scaled_dot_product_attention(a, b, c, causal=True), (q, k, v)
    )


def copy_into_torch(ours: MultiHeadAttention) -> torch.nn.MultiheadAttention:
    d = ours.d_model
    ref = torch.nn.MultiheadAttention(d, ours.h, batch_first=True)
    with torch.no_grad():
        ref.in_proj_weight.copy_(
            torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight])
        )
        ref.in_proj_bias.copy_(torch.cat([ours.q_proj.bias, ours.k_proj.bias, ours.v_proj.bias]))
        ref.out_proj.weight.copy_(ours.out_proj.weight)
        ref.out_proj.bias.copy_(ours.out_proj.bias)
    return ref


@pytest.mark.parametrize("n_heads", [1, 2, 4])
def test_multihead_matches_torch_module(n_heads):
    torch.manual_seed(5)
    ours = MultiHeadAttention(16, n_heads)
    ref = copy_into_torch(ours)
    x = torch.randn(3, 7, 16)
    out, w = ours(x, return_weights=True)
    ref_out, ref_w = ref(x, x, x, average_attn_weights=False)
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(w, ref_w)
    # Causal self-attention.
    causal_mask = torch.triu(torch.ones(7, 7, dtype=torch.bool), 1)  # torch: True = BLOCK
    torch.testing.assert_close(ours(x, causal=True), ref(x, x, x, attn_mask=causal_mask)[0])
    # Cross-attention with key padding.
    mem = torch.randn(3, 5, 16)
    keep = torch.ones(3, 5, dtype=torch.bool)
    keep[0, 3:] = False
    torch.testing.assert_close(ours(x, mem, mask=keep), ref(x, mem, mem, key_padding_mask=~keep)[0])


def test_self_attention_is_permutation_equivariant():
    """Without positional information, shuffling the tokens just shuffles the outputs."""
    torch.manual_seed(6)
    mha = MultiHeadAttention(8, 2)
    x = torch.randn(1, 6, 8)
    perm = torch.randperm(6)
    torch.testing.assert_close(mha(x)[:, perm], mha(x[:, perm]))


def test_invalid_head_count():
    with pytest.raises(ValueError):
        MultiHeadAttention(10, 3)
