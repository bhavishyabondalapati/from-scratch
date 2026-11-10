from pathlib import Path

import numpy as np
import pytest
import torch

from s2_math.gradient_check.impl import check_gradient

from .impl import GRU, LSTM, RNN, gradient_norm_through_time

PAIRS = [(RNN, torch.nn.RNN), (LSTM, torch.nn.LSTM), (GRU, torch.nn.GRU)]


def torch_copy(ours, torch_cls, D, H):
    ref = torch_cls(D, H).double()
    with torch.no_grad():
        for name in ("W_ih", "W_hh", "b_ih", "b_hh"):
            torch_name = name.replace("W_", "weight_").replace("b_", "bias_") + "_l0"
            getattr(ref, torch_name).copy_(torch.tensor(ours.params[name]))
    return ref


@pytest.mark.parametrize("ours_cls, torch_cls", PAIRS, ids=["rnn", "lstm", "gru"])
def test_forward_and_bptt_match_pytorch(ours_cls, torch_cls, rng):
    T, N, D, H = 6, 3, 4, 5
    ours = ours_cls(D, H, seed=1)
    ref = torch_copy(ours, torch_cls, D, H)
    x = rng.normal(size=(T, N, D))
    dhs = rng.normal(size=(T, N, H))
    hs = ours.forward(x)
    dx = ours.backward(dhs)
    xt = torch.tensor(x, requires_grad=True)
    out, _ = ref(xt)
    out.backward(torch.tensor(dhs))
    np.testing.assert_allclose(hs, out.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(dx, xt.grad.numpy(), atol=1e-10)
    for name in ("W_ih", "W_hh", "b_ih", "b_hh"):
        torch_name = name.replace("W_", "weight_").replace("b_", "bias_") + "_l0"
        np.testing.assert_allclose(
            ours.grads[name], getattr(ref, torch_name).grad.numpy(), atol=1e-10
        )


@pytest.mark.parametrize("cls", [RNN, LSTM, GRU])
def test_bptt_passes_finite_difference_check(cls, rng):
    layer = cls(3, 4, seed=2)
    x = rng.normal(size=(5, 2, 3))
    dhs = rng.normal(size=(5, 2, 4))

    def loss(_=None):
        return float(np.sum(layer.forward(x) * dhs))

    layer.forward(x)
    dx = layer.backward(dhs)
    analytic = {k: v.copy() for k, v in layer.grads.items()}
    ok, err = check_gradient(lambda z: float(np.sum(layer.forward(z) * dhs)), dx, x)
    assert ok, err
    for name, g in analytic.items():
        ok, err = check_gradient(loss, g, layer.params[name])
        assert ok, (name, err)


def test_lstm_gradient_survives_longer_than_rnn():
    T = 60
    rnn = RNN(8, 32, seed=0)
    lstm = LSTM(8, 32, seed=0)
    lstm.params["b_ih"][32:64] += 3.0  # forget-gate bias > 0: "remember by default" (common trick)
    g_rnn = gradient_norm_through_time(rnn, T)
    g_lstm = gradient_norm_through_time(lstm, T)
    # Ratio of influence of the first input vs the last input on the final state.
    assert g_rnn[0] / g_rnn[-1] < 1e-6  # vanished
    assert g_lstm[0] / g_lstm[-1] > 1e-3  # still there

    import matplotlib.pyplot as plt

    gru = gradient_norm_through_time(GRU(8, 32, seed=0), T)
    fig, ax = plt.subplots(figsize=(5, 3))
    for name, g in [("RNN", g_rnn), ("LSTM (forget bias 3)", g_lstm), ("GRU", gru)]:
        ax.semilogy(range(T), g / g[-1], label=name)
    ax.set(
        xlabel="input time step t",
        ylabel="|dh_T/dx_t| (relative)",
        title="Gradient flow back through 60 steps",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "gradient_flow.png", dpi=110)
    plt.close(fig)


def test_initial_state_gradients(rng):
    layer = LSTM(2, 3, seed=0)
    x = rng.normal(size=(4, 1, 2))
    h0, c0 = rng.normal(size=(1, 3)), rng.normal(size=(1, 3))
    dhs = rng.normal(size=(4, 1, 3))
    layer.forward(x, h0, c0)
    layer.backward(dhs)
    ok, err = check_gradient(lambda h: float(np.sum(layer.forward(x, h, c0) * dhs)), layer.dh0, h0)
    assert ok, err
    layer.forward(x, h0, c0)
    layer.backward(dhs)
    ok, err = check_gradient(lambda c: float(np.sum(layer.forward(x, h0, c) * dhs)), layer.dc0, c0)
    assert ok, err
