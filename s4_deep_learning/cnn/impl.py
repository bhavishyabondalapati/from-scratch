"""Convolutional layers in NumPy: conv2d via im2col, max/avg pooling, and a small CNN.

Shapes follow PyTorch: images are (N, C, H, W); conv weights are (C_out, C_in, kH, kW).
"""

from __future__ import annotations

import math

import numpy as np

from s4_deep_learning.mlp.impl import Layer, Linear, ReLU, Sequential


def output_size(size: int, kernel: int, stride: int, pad: int) -> int:
    return (size + 2 * pad - kernel) // stride + 1


def _patch_indices(C: int, H: int, W: int, kh: int, kw: int, stride: int, pad: int):
    """Index arrays (k, i, j) such that x_padded[:, k, i, j] has shape
    (N, C*kh*kw, out_h*out_w): one column per output position, holding the
    receptive field that the kernel sees there."""
    out_h, out_w = output_size(H, kh, stride, pad), output_size(W, kw, stride, pad)
    i0 = np.tile(np.repeat(np.arange(kh), kw), C)  # row offset inside the patch
    j0 = np.tile(np.arange(kw), kh * C)  # column offset inside the patch
    i1 = stride * np.repeat(np.arange(out_h), out_w)  # patch's top row in the image
    j1 = stride * np.tile(np.arange(out_w), out_h)
    i = i0[:, None] + i1[None, :]
    j = j0[:, None] + j1[None, :]
    k = np.repeat(np.arange(C), kh * kw)[:, None]  # channel of each patch row
    return k, i, j, out_h, out_w


def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1, pad: int = 0) -> np.ndarray:
    """Unroll every receptive field into a column. Convolution becomes ONE matmul."""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    k, i, j, _, _ = _patch_indices(C, H, W, kh, kw, stride, pad)
    return xp[:, k, i, j]  # (N, C*kh*kw, out_h*out_w)


def col2im(
    cols: np.ndarray, x_shape: tuple, kh: int, kw: int, stride: int = 1, pad: int = 0
) -> np.ndarray:
    """Inverse scatter of im2col: add every column entry back to the pixel it came
    from. Overlapping patches touch the same pixel several times, so we ADD."""
    N, C, H, W = x_shape
    xp = np.zeros((N, C, H + 2 * pad, W + 2 * pad))
    k, i, j, _, _ = _patch_indices(C, H, W, kh, kw, stride, pad)
    np.add.at(xp, (slice(None), k, i, j), cols)
    return xp[:, :, pad : pad + H, pad : pad + W]


def conv2d_naive(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int = 1, pad: int = 0
) -> np.ndarray:
    """Direct definition with loops: out[n, f, r, c] = sum over the window of x * w + b[f].
    Slow; used only to test the fast version."""
    N, C, H, W = x.shape
    F, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh, ow = output_size(H, kh, stride, pad), output_size(W, kw, stride, pad)
    out = np.zeros((N, F, oh, ow))
    for r in range(oh):
        for c in range(ow):
            window = xp[:, :, r * stride : r * stride + kh, c * stride : c * stride + kw]
            out[:, :, r, c] = np.tensordot(window, w, axes=([1, 2, 3], [1, 2, 3])) + b
    return out


class Conv2d(Layer):
    """Forward:  out = W_flat @ cols + b,  W_flat: (F, C*kh*kw), cols: (N, C*kh*kw, L).
    Backward:
        dW = sum_n  G_n @ cols_n^T         (how each weight affected every output)
        db = sum over N and positions of G
        dx = col2im(W_flat^T @ G)          (send gradient back to every pixel that was used)
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel: int,
        stride: int = 1,
        pad: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng(0)
        fan_in = c_in * kernel * kernel
        self.params = {
            "W": rng.normal(
                0, math.sqrt(2 / fan_in), size=(c_out, c_in, kernel, kernel)
            ),  # He init
            "b": np.zeros(c_out),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.k, self.stride, self.pad = kernel, stride, pad

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, _, H, W = x.shape
        F = self.params["W"].shape[0]
        self.x_shape = x.shape
        self.cols = im2col(x, self.k, self.k, self.stride, self.pad)
        out = self.params["W"].reshape(F, -1) @ self.cols + self.params["b"][:, None]
        oh = output_size(H, self.k, self.stride, self.pad)
        ow = output_size(W, self.k, self.stride, self.pad)
        return out.reshape(N, F, oh, ow)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        N, F = grad_out.shape[:2]
        G = grad_out.reshape(N, F, -1)  # (N, F, L)
        W_flat = self.params["W"].reshape(F, -1)
        self.grads["W"] = np.einsum("nfl,nkl->fk", G, self.cols).reshape(self.params["W"].shape)
        self.grads["b"] = G.sum(axis=(0, 2))
        dcols = W_flat.T @ G  # (N, C*kh*kw, L)
        return col2im(dcols, self.x_shape, self.k, self.k, self.stride, self.pad)


class MaxPool2d(Layer):
    """Keep the largest value in each window; the gradient goes only to that position."""

    def __init__(self, kernel: int = 2, stride: int | None = None) -> None:
        self.k, self.stride = kernel, stride or kernel

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        self.x_shape = x.shape
        # Treat every channel as a separate 1-channel image so windows don't mix channels.
        cols = im2col(x.reshape(N * C, 1, H, W), self.k, self.k, self.stride)
        self.argmax = cols.argmax(axis=1)  # (N*C, L)
        self.n_cols = cols.shape
        out = cols.max(axis=1)
        oh, ow = output_size(H, self.k, self.stride, 0), output_size(W, self.k, self.stride, 0)
        return out.reshape(N, C, oh, ow)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        N, C, H, W = self.x_shape
        dcols = np.zeros(self.n_cols)
        rows = np.arange(self.n_cols[0])[:, None]
        pos = np.arange(self.n_cols[2])[None, :]
        dcols[rows, self.argmax, pos] = grad_out.reshape(N * C, -1)
        return col2im(dcols, (N * C, 1, H, W), self.k, self.k, self.stride).reshape(self.x_shape)


class AvgPool2d(Layer):
    def __init__(self, kernel: int = 2, stride: int | None = None) -> None:
        self.k, self.stride = kernel, stride or kernel

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        self.x_shape = x.shape
        cols = im2col(x.reshape(N * C, 1, H, W), self.k, self.k, self.stride)
        self.n_cols = cols.shape
        oh, ow = output_size(H, self.k, self.stride, 0), output_size(W, self.k, self.stride, 0)
        return cols.mean(axis=1).reshape(N, C, oh, ow)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        N, C, H, W = self.x_shape
        g = grad_out.reshape(N * C, 1, -1) / (self.k * self.k)
        dcols = np.broadcast_to(g, self.n_cols)
        return col2im(dcols, (N * C, 1, H, W), self.k, self.k, self.stride).reshape(self.x_shape)


class Flatten(Layer):
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, grad_out):
        return grad_out.reshape(self.shape)


def small_cnn(seed: int = 0) -> Sequential:
    """For 8x8 single-channel digits:
    conv3x3(1->16, pad 1) -> ReLU -> maxpool 2 -> conv3x3(16->32, pad 1) -> ReLU -> maxpool 2
    -> flatten (32*2*2) -> linear 128 -> ReLU -> linear 10
    """
    rng = np.random.default_rng(seed)
    return Sequential(
        Conv2d(1, 16, 3, pad=1, rng=rng),
        ReLU(),
        MaxPool2d(2),
        Conv2d(16, 32, 3, pad=1, rng=rng),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(32 * 2 * 2, 128, rng=rng),
        ReLU(),
        Linear(128, 10, rng=rng),
    )
