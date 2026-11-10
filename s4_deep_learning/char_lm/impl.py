"""Character-level language model: embedding -> LSTM -> linear -> softmax, all NumPy.

Trained with truncated backprop through time, Adam, and gradient-norm clipping.
"""

from __future__ import annotations

import math

import numpy as np

from s2_math.softmax_logsumexp.impl import log_softmax, softmax
from s4_deep_learning.rnn_lstm_gru.impl import LSTM


class Vocab:
    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self.index = {c: i for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        return np.array([self.index[c] for c in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.chars[i] for i in ids)


class CharLSTM:
    """Predict the next character from all previous ones.

    tokens (T, N) -> embeddings E[tokens] (T, N, D) -> LSTM hidden states (T, N, H)
    -> logits (T, N, V) = h W_out + b_out
    """

    def __init__(
        self, vocab_size: int, embed_dim: int = 32, hidden: int = 128, seed: int = 0
    ) -> None:
        rng = np.random.default_rng(seed)
        self.lstm = LSTM(embed_dim, hidden, seed=seed)
        self.lstm.params["b_ih"][hidden : 2 * hidden] += (
            1.0  # forget-gate bias: remember by default
        )
        self.params = {
            "E": rng.normal(0, 0.1, (vocab_size, embed_dim)),
            "W_out": rng.normal(0, 1 / math.sqrt(hidden), (hidden, vocab_size)),
            "b_out": np.zeros(vocab_size),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def all_params(self):
        """(dict, key) pairs for every trainable array, including the LSTM's."""
        for k in self.params:
            yield self.params, self.grads, k
        for k in self.lstm.params:
            yield self.lstm.params, self.lstm.grads, k

    def forward(self, tokens: np.ndarray, h0=None, c0=None) -> np.ndarray:
        self.tokens = tokens
        hs = self.lstm.forward(self.params["E"][tokens], h0, c0)
        self.hs = hs
        return hs @ self.params["W_out"] + self.params["b_out"]

    def loss(self, tokens: np.ndarray, targets: np.ndarray, h0=None, c0=None) -> float:
        """Mean cross-entropy (nats per character) and cache for backward."""
        logits = self.forward(tokens, h0, c0)
        T, N, V = logits.shape
        self.flat_logits, self.flat_targets = logits.reshape(-1, V), targets.reshape(-1)
        logp = log_softmax(self.flat_logits)
        return float(-logp[np.arange(T * N), self.flat_targets].mean())

    def backward(self) -> None:
        T, N, H = self.hs.shape
        dlogits = softmax(self.flat_logits)
        dlogits[np.arange(len(self.flat_targets)), self.flat_targets] -= 1
        dlogits /= len(self.flat_targets)
        flat_h = self.hs.reshape(-1, H)
        self.grads["W_out"] = flat_h.T @ dlogits
        self.grads["b_out"] = dlogits.sum(axis=0)
        dhs = (dlogits @ self.params["W_out"].T).reshape(T, N, H)
        dembed = self.lstm.backward(dhs)
        self.grads["E"] = np.zeros_like(self.params["E"])
        np.add.at(self.grads["E"], self.tokens, dembed)  # each row gets the sum over its uses

    def sample(
        self, vocab: Vocab, prompt: str, n: int, temperature: float = 1.0, seed: int = 0
    ) -> str:
        """Generate n characters after `prompt`. Lower temperature = more conservative."""
        rng = np.random.default_rng(seed)
        h = c = np.zeros((1, self.lstm.H))
        ids = list(vocab.encode(prompt))
        # Feed the prompt, then keep feeding back our own predictions one step at a time.
        logits = self.forward(np.array(ids)[:, None], h, c)[-1, 0]
        h, c = self.lstm.hs[-1], self.lstm.cs[-1]
        out = []
        for _ in range(n):
            if temperature == 0:
                nxt = int(np.argmax(logits))
            else:
                nxt = int(rng.choice(len(vocab), p=softmax(logits / temperature)))
            out.append(nxt)
            logits = self.forward(np.array([[nxt]]), h, c)[-1, 0]
            h, c = self.lstm.hs[-1], self.lstm.cs[-1]
        return vocab.decode(out)


class Adam:
    def __init__(
        self, model: CharLSTM, lr: float = 3e-3, betas=(0.9, 0.999), eps: float = 1e-8
    ) -> None:
        self.model, self.lr, (self.b1, self.b2), self.eps = model, lr, betas, eps
        self.state: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        self.t = 0

    def step(self, clip: float | None = 5.0) -> float:
        """Clip the GLOBAL gradient norm (protects against exploding gradients), then update.
        Returns the norm before clipping."""
        triples = list(self.model.all_params())
        norm = math.sqrt(sum(float(np.sum(g[k] ** 2)) for _, g, k in triples))
        scale = clip / norm if clip and norm > clip else 1.0
        self.t += 1
        for p, g, k in triples:
            grad = g[k] * scale
            m, v = self.state.get((id(p), k), (np.zeros_like(grad), np.zeros_like(grad)))
            m = self.b1 * m + (1 - self.b1) * grad
            v = self.b2 * v + (1 - self.b2) * grad**2
            self.state[(id(p), k)] = (m, v)
            m_hat, v_hat = m / (1 - self.b1**self.t), v / (1 - self.b2**self.t)
            p[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return norm


def get_batch(data: np.ndarray, seq_len: int, batch: int, rng: np.random.Generator):
    """Random windows; targets are inputs shifted by one character. Shapes (T, N)."""
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch)
    x = np.stack([data[s : s + seq_len] for s in starts], axis=1)
    y = np.stack([data[s + 1 : s + seq_len + 1] for s in starts], axis=1)
    return x, y


def unigram_loss(train: np.ndarray, val: np.ndarray, vocab_size: int) -> float:
    """Baseline: predict every character from overall character frequencies."""
    counts = np.bincount(train, minlength=vocab_size) + 1.0
    logp = np.log(counts / counts.sum())
    return float(-logp[val].mean())


def evaluate(
    model: CharLSTM,
    data: np.ndarray,
    seq_len: int = 100,
    batch: int = 32,
    n_batches: int = 10,
    seed: int = 123,
) -> float:
    rng = np.random.default_rng(seed)
    return float(
        np.mean([model.loss(*get_batch(data, seq_len, batch, rng)) for _ in range(n_batches)])
    )


def train(
    text: str,
    steps: int = 1500,
    seq_len: int = 64,
    batch: int = 32,
    hidden: int = 128,
    lr: float = 3e-3,
    eval_every: int = 100,
    seed: int = 0,
):
    vocab = Vocab(text)
    data = vocab.encode(text)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    model = CharLSTM(len(vocab), hidden=hidden, seed=seed)
    opt = Adam(model, lr)
    rng = np.random.default_rng(seed)
    history = {"step": [], "train": [], "val": []}
    running = None
    for step in range(1, steps + 1):
        loss = model.loss(*get_batch(train_data, seq_len, batch, rng))
        model.backward()
        opt.step(clip=5.0)
        running = loss if running is None else 0.95 * running + 0.05 * loss
        if step % eval_every == 0 or step == steps:
            history["step"].append(step)
            history["train"].append(running)
            history["val"].append(
                evaluate(model, val_data, seq_len=min(seq_len, len(val_data) - 2))
            )
    baseline = unigram_loss(train_data, val_data, len(vocab))
    return model, vocab, history, baseline
