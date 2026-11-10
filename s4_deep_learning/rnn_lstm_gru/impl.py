"""Vanilla RNN, LSTM and GRU layers with hand-written backprop through time (BPTT), in NumPy.

Inputs are (T, N, D): time steps, batch, features. Each layer returns all hidden
states (T, N, H). Weight names and gate orders follow PyTorch (nn.RNN / nn.LSTM /
nn.GRU) so the tests can copy weights across and compare exactly.
"""

from __future__ import annotations

import math

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + e), e / (1 + e))


class _Recurrent:
    n_gates = 1

    def __init__(self, input_size: int, hidden_size: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        k = 1 / math.sqrt(hidden_size)  # PyTorch's default: U(-1/sqrt(H), 1/sqrt(H))
        G = self.n_gates * hidden_size
        self.params = {
            "W_ih": rng.uniform(-k, k, (G, input_size)),
            "W_hh": rng.uniform(-k, k, (G, hidden_size)),
            "b_ih": rng.uniform(-k, k, G),
            "b_hh": rng.uniform(-k, k, G),
        }
        self.grads = {name: np.zeros_like(p) for name, p in self.params.items()}
        self.H = hidden_size

    def _zero_grads(self) -> None:
        for g in self.grads.values():
            g[...] = 0.0

    def _accumulate_linear(
        self, x_t: np.ndarray, h_prev: np.ndarray, d_pre_x: np.ndarray, d_pre_h: np.ndarray
    ):
        """Grads for pre_x = x W_ih^T + b_ih and pre_h = h W_hh^T + b_hh."""
        self.grads["W_ih"] += d_pre_x.T @ x_t
        self.grads["b_ih"] += d_pre_x.sum(axis=0)
        self.grads["W_hh"] += d_pre_h.T @ h_prev
        self.grads["b_hh"] += d_pre_h.sum(axis=0)
        return d_pre_x @ self.params["W_ih"], d_pre_h @ self.params["W_hh"]


class RNN(_Recurrent):
    """h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)."""

    def forward(self, x: np.ndarray, h0: np.ndarray | None = None) -> np.ndarray:
        T, N, _ = x.shape
        h = np.zeros((N, self.H)) if h0 is None else h0
        self.x, self.hs = x, [h]
        for t in range(T):
            h = np.tanh(
                x[t] @ self.params["W_ih"].T
                + self.params["b_ih"]
                + h @ self.params["W_hh"].T
                + self.params["b_hh"]
            )
            self.hs.append(h)
        return np.stack(self.hs[1:])

    def backward(self, dhs: np.ndarray) -> np.ndarray:
        """dhs[t] = dLoss/dh_t coming from outside (e.g. an output layer at each step).

        Walking backwards, the gradient reaching h_t is its own outside gradient
        PLUS what flows back from h_{t+1} through W_hh. That second term is
        multiplied by W_hh^T * tanh' at every step, which is why plain RNN
        gradients vanish or explode over long sequences.
        """
        self._zero_grads()
        dx = np.zeros_like(self.x)
        dh_next = np.zeros_like(self.hs[0])
        for t in reversed(range(len(dhs))):
            h, h_prev = self.hs[t + 1], self.hs[t]
            dpre = (dhs[t] + dh_next) * (1 - h**2)
            dx[t], dh_next = self._accumulate_linear(self.x[t], h_prev, dpre, dpre)
        self.dh0 = dh_next
        return dx


class LSTM(_Recurrent):
    """Gates (PyTorch order i, f, g, o) from z = W_ih x + b_ih + W_hh h + b_hh:
        i = sigmoid  (input gate: how much new info to write)
        f = sigmoid  (forget gate: how much old memory to keep)
        g = tanh     (candidate values)
        o = sigmoid  (output gate: how much memory to expose)
        c_t = f * c_{t-1} + i * g        h_t = o * tanh(c_t)

    The cell state c is updated by ADDITION, so its gradient flows back through
    time multiplied only by f (no squashing weight matrix): the "constant error
    carousel" that lets LSTMs remember over long spans.
    """

    n_gates = 4

    def forward(
        self, x: np.ndarray, h0: np.ndarray | None = None, c0: np.ndarray | None = None
    ) -> np.ndarray:
        T, N, _ = x.shape
        H = self.H
        h = np.zeros((N, H)) if h0 is None else h0
        c = np.zeros((N, H)) if c0 is None else c0
        self.x, self.hs, self.cs, self.cache = x, [h], [c], []
        for t in range(T):
            z = (
                x[t] @ self.params["W_ih"].T
                + self.params["b_ih"]
                + h @ self.params["W_hh"].T
                + self.params["b_hh"]
            )
            i, f = _sigmoid(z[:, :H]), _sigmoid(z[:, H : 2 * H])
            g, o = np.tanh(z[:, 2 * H : 3 * H]), _sigmoid(z[:, 3 * H :])
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            self.cache.append((i, f, g, o, tc))
            self.hs.append(h)
            self.cs.append(c)
        return np.stack(self.hs[1:])

    def backward(self, dhs: np.ndarray) -> np.ndarray:
        self._zero_grads()
        dx = np.zeros_like(self.x)
        dh_next = np.zeros_like(self.hs[0])
        dc_next = np.zeros_like(self.cs[0])
        for t in reversed(range(len(dhs))):
            i, f, g, o, tc = self.cache[t]
            dh = dhs[t] + dh_next
            dc = dc_next + dh * o * (1 - tc**2)  # through h = o * tanh(c), plus from c_{t+1}
            di, df, dg, do = dc * g, dc * self.cs[t], dc * i, dh * tc
            dz = np.concatenate(
                [di * i * (1 - i), df * f * (1 - f), dg * (1 - g**2), do * o * (1 - o)], axis=1
            )
            dx[t], dh_next = self._accumulate_linear(self.x[t], self.hs[t], dz, dz)
            dc_next = dc * f  # the cell-state highway: only scaled by the forget gate
        self.dh0, self.dc0 = dh_next, dc_next
        return dx


class GRU(_Recurrent):
    """PyTorch GRU (gate order r, z, n):
        r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)        reset gate
        z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)        update gate
        n = tanh(W_in x + b_in + r * (W_hn h + b_hn))     candidate
        h_t = (1 - z) * n + z * h_{t-1}
    Like the LSTM it has an additive path (z * h_{t-1}) but no separate cell state.
    """

    n_gates = 3

    def forward(self, x: np.ndarray, h0: np.ndarray | None = None) -> np.ndarray:
        T, N, _ = x.shape
        H = self.H
        h = np.zeros((N, H)) if h0 is None else h0
        self.x, self.hs, self.cache = x, [h], []
        for t in range(T):
            ax = x[t] @ self.params["W_ih"].T + self.params["b_ih"]
            ah = h @ self.params["W_hh"].T + self.params["b_hh"]
            r = _sigmoid(ax[:, :H] + ah[:, :H])
            z = _sigmoid(ax[:, H : 2 * H] + ah[:, H : 2 * H])
            n = np.tanh(ax[:, 2 * H :] + r * ah[:, 2 * H :])
            h = (1 - z) * n + z * h
            self.cache.append((r, z, n, ah[:, 2 * H :]))
            self.hs.append(h)
        return np.stack(self.hs[1:])

    def backward(self, dhs: np.ndarray) -> np.ndarray:
        self._zero_grads()
        dx = np.zeros_like(self.x)
        dh_next = np.zeros_like(self.hs[0])
        for t in reversed(range(len(dhs))):
            r, z, n, ah_n = self.cache[t]
            h_prev = self.hs[t]
            dh = dhs[t] + dh_next
            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dn_pre = dn * (1 - n**2)
            dr = dn_pre * ah_n
            dr_pre, dz_pre = dr * r * (1 - r), dz * z * (1 - z)
            d_ax = np.concatenate([dr_pre, dz_pre, dn_pre], axis=1)
            d_ah = np.concatenate([dr_pre, dz_pre, dn_pre * r], axis=1)
            dx[t], dh_prev_lin = self._accumulate_linear(self.x[t], h_prev, d_ax, d_ah)
            dh_next = dh_prev_lin + dh * z  # through W_hh, plus the direct z * h_{t-1} path
        self.dh0 = dh_next
        return dx


def gradient_norm_through_time(layer: _Recurrent, T: int = 50, seed: int = 0) -> np.ndarray:
    """|dL/dx_t| for L = sum(h_T): how much the LAST output depends on each earlier input."""
    x = np.random.default_rng(seed).normal(size=(T, 1, layer.params["W_ih"].shape[1]))
    hs = layer.forward(x)
    dhs = np.zeros_like(hs)
    dhs[-1] = 1.0
    return np.linalg.norm(layer.backward(dhs)[:, 0, :], axis=1)
