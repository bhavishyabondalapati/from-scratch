import math
from pathlib import Path

import pytest
import torch

from .impl import apply_rope, rope_as_complex, rope_frequencies, sinusoidal_encoding, wavelengths


def test_sinusoidal_matches_formula_and_has_unit_pairs():
    pe = sinusoidal_encoding(50, 16)
    pos, i = 7, 3
    assert pe[pos, 2 * i].item() == pytest.approx(math.sin(pos / 10000 ** (2 * i / 16)), abs=1e-6)
    assert pe[pos, 2 * i + 1].item() == pytest.approx(math.cos(pos / 10000 ** (2 * i / 16)), abs=1e-6)
    pair_norms = pe[:, 0::2] ** 2 + pe[:, 1::2] ** 2
    torch.testing.assert_close(pair_norms, torch.ones_like(pair_norms))
    assert torch.allclose(pe[0, 0::2], torch.zeros(8)) and torch.allclose(pe[0, 1::2], torch.ones(8))
    w = wavelengths(16)
    assert w[0].item() == pytest.approx(2 * math.pi) and torch.all(w[1:] > w[:-1])


def test_sinusoidal_shift_is_a_linear_map():
    """PE(pos + k) = R_k PE(pos) for a fixed block-rotation R_k: relative positions are easy to read."""
    d, k = 8, 5
    pe = sinusoidal_encoding(100, d).double()
    freqs = 10000 ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)
    R = torch.zeros(d, d, dtype=torch.float64)
    for j, f in enumerate(freqs):
        c, s = math.cos(k * f), math.sin(k * f)
        R[2 * j : 2 * j + 2, 2 * j : 2 * j + 2] = torch.tensor([[c, s], [-s, c]], dtype=torch.float64)
    torch.testing.assert_close(pe[k:] , pe[:-k] @ R, atol=1e-5, rtol=0)


def test_nearby_positions_are_more_similar():
    pe = sinusoidal_encoding(200, 64)
    sims = pe[100] @ pe.T
    assert sims[100] > sims[101] > sims[110] and sims[101] > sims[150]


def test_rope_matches_complex_number_reference():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 10, 8)
    cos, sin = rope_frequencies(8, 64)
    torch.testing.assert_close(apply_rope(x, cos, sin), rope_as_complex(x, torch.arange(10)), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        apply_rope(x, cos, sin, offset=20), rope_as_complex(x, torch.arange(20, 30)), atol=1e-5, rtol=1e-5
    )


def test_rope_preserves_norms_and_position_zero():
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    cos, sin = rope_frequencies(16, 32)
    y = apply_rope(x, cos, sin)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1))
    torch.testing.assert_close(y[0], x[0])  # position 0: rotation by angle 0


def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(2)
    d = 16
    cos, sin = rope_frequencies(d, 128)
    q, k = torch.randn(d), torch.randn(d)

    def score(m, n):
        qm = apply_rope(q.expand(m + 1, d), cos, sin)[m]
        kn = apply_rope(k.expand(n + 1, d), cos, sin)[n]
        return float(qm @ kn)

    assert score(5, 2) == pytest.approx(score(50, 47), abs=1e-4)  # both offset 3
    assert score(10, 10) == pytest.approx(score(0, 0), abs=1e-4) == pytest.approx(float(q @ k), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-3)
    with pytest.raises(ValueError):
        rope_frequencies(7, 10)


def test_save_figure():
    import matplotlib.pyplot as plt

    pe = sinusoidal_encoding(100, 64)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(pe.T, aspect="auto", cmap="RdBu")
    ax.set(xlabel="position", ylabel="dimension", title="Sinusoidal positional encoding (d=64)")
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "sinusoidal.png", dpi=110)
    plt.close(fig)
