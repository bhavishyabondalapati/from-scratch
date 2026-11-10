"""word2vec: skip-gram with negative sampling (SGNS) in NumPy, plus similarity and analogy tools."""

from __future__ import annotations

from collections import Counter

import numpy as np

from s2_math.softmax_logsumexp.impl import log_sigmoid, sigmoid

# --- A synthetic corpus with known structure (shared with the GloVe folder) ------------

NOUNS = {  # word -> (gender, status, age)
    "king": ("m", "royal", "adult"), "queen": ("f", "royal", "adult"),
    "prince": ("m", "royal", "child"), "princess": ("f", "royal", "child"),
    "man": ("m", "common", "adult"), "woman": ("f", "common", "adult"),
    "boy": ("m", "common", "child"), "girl": ("f", "common", "child"),
}  # fmt: skip
CONTEXT = {
    "m": ["he", "his", "him", "himself"],
    "f": ["she", "her", "hers", "herself"],
    "royal": ["crown", "palace", "throne", "castle", "rules"],
    "common": ["farm", "village", "market", "works", "cottage"],
    "adult": ["old", "wise", "married", "taxes"],
    "child": ["young", "plays", "school", "toys"],
}
FILLER = ["the", "a", "and", "then", "today", "there", "with", "very"]


def synthetic_corpus(n_sentences: int = 4000, seed: int = 0) -> list[list[str]]:
    """Each sentence mentions one noun plus context words drawn from its three attributes,
    so a noun's co-occurrence profile is (roughly) the SUM of its attributes' profiles."""
    rng = np.random.default_rng(seed)
    nouns = list(NOUNS)
    sentences = []
    for _ in range(n_sentences):
        noun = nouns[rng.integers(len(nouns))]
        words = [noun] + [rng.choice(CONTEXT[a]) for a in NOUNS[noun] for _ in range(2)]
        words += list(rng.choice(FILLER, size=3))
        rng.shuffle(words)
        sentences.append([str(w) for w in words])
    return sentences


# --- Vocabulary and training pairs --------------------------------------------------------


class Vocab:
    def __init__(self, sentences: list[list[str]], min_count: int = 1) -> None:
        counts = Counter(w for s in sentences for w in s)
        self.words = sorted(w for w, c in counts.items() if c >= min_count)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.counts = np.array([counts[w] for w in self.words], dtype=float)

    def __len__(self) -> int:
        return len(self.words)

    def encode(self, sentence: list[str]) -> list[int]:
        return [self.index[w] for w in sentence if w in self.index]


def skipgram_pairs(sentences: list[list[int]], window: int, rng: np.random.Generator) -> np.ndarray:
    """(center, context) pairs. Like the original code, each center uses a random window
    size in 1..window, which weights nearby words more than distant ones."""
    pairs = []
    for s in sentences:
        for i, c in enumerate(s):
            w = int(rng.integers(1, window + 1))
            for j in range(max(0, i - w), min(len(s), i + w + 1)):
                if j != i:
                    pairs.append((c, s[j]))
    return np.array(pairs)


def noise_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    """Negatives are drawn from unigram^0.75: flatter than raw frequency, so rare words
    are sampled a bit more often than they occur."""
    p = counts**power
    return p / p.sum()


# --- The model --------------------------------------------------------------------------------


class SkipGramNS:
    """Two embedding tables: W_in for words as CENTERS, W_out for words as CONTEXTS.

    For a true pair (c, o) and k sampled negatives n_1..n_k, the loss is
        -log sigma(u_o . v_c) - sum_i log sigma(-u_{n_i} . v_c)
    i.e. a logistic regression "real pair vs noise pair", which replaces the
    expensive softmax over the whole vocabulary.
    """

    def __init__(self, vocab_size: int, dim: int = 50, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-0.5 / dim, 0.5 / dim, (vocab_size, dim))
        self.W_out = np.zeros((vocab_size, dim))  # zero init, as in the original code

    def loss_and_grads(self, centers: np.ndarray, contexts: np.ndarray, negatives: np.ndarray):
        """centers, contexts: (B,); negatives: (B, k). Returns mean loss and gradients
        w.r.t. the rows used: (dv (B, d), du_pos (B, d), du_neg (B, k, d))."""
        v = self.W_in[centers]  # (B, d)
        u_pos = self.W_out[contexts]  # (B, d)
        u_neg = self.W_out[negatives]  # (B, k, d)
        s_pos = np.sum(u_pos * v, axis=1)  # (B,)
        s_neg = np.einsum("bkd,bd->bk", u_neg, v)  # (B, k)
        loss = -(log_sigmoid(s_pos) + log_sigmoid(-s_neg).sum(axis=1)).mean()
        B = len(centers)
        g_pos = (sigmoid(s_pos) - 1) / B  # d loss / d s_pos
        g_neg = sigmoid(s_neg) / B  # d loss / d s_neg
        dv = g_pos[:, None] * u_pos + np.einsum("bk,bkd->bd", g_neg, u_neg)
        du_pos = g_pos[:, None] * v
        du_neg = g_neg[:, :, None] * v[:, None, :]
        return float(loss), dv, du_pos, du_neg

    def sgd_step(self, centers, contexts, negatives, lr: float) -> float:
        loss, dv, du_pos, du_neg = self.loss_and_grads(centers, contexts, negatives)
        B = len(centers)
        # np.add.at: the same word can appear many times in one batch.
        np.add.at(self.W_in, centers, -lr * B * dv)
        np.add.at(self.W_out, contexts, -lr * B * du_pos)
        np.add.at(self.W_out, negatives, -lr * B * du_neg)
        return loss


def train_word2vec(
    sentences: list[list[str]],
    dim: int = 30,
    window: int = 4,
    negatives: int = 5,
    epochs: int = 5,
    lr: float = 0.025,
    batch: int = 256,
    seed: int = 0,
):
    vocab = Vocab(sentences)
    rng = np.random.default_rng(seed)
    ids = [vocab.encode(s) for s in sentences]
    noise = noise_distribution(vocab.counts)
    model = SkipGramNS(len(vocab), dim, seed)
    history = []
    for epoch in range(epochs):
        pairs = skipgram_pairs(ids, window, rng)
        rng.shuffle(pairs)
        losses = []
        for start in range(0, len(pairs), batch):
            chunk = pairs[start : start + batch]
            neg = rng.choice(len(vocab), size=(len(chunk), negatives), p=noise)
            # Linearly decay the learning rate over training, as the original implementation does.
            progress = (epoch + start / len(pairs)) / epochs
            losses.append(
                model.sgd_step(chunk[:, 0], chunk[:, 1], neg, lr * max(1 - progress, 1e-4))
            )
        history.append(float(np.mean(losses)))
    return model, vocab, history


# --- Using the embeddings --------------------------------------------------------------------


def normalize(E: np.ndarray) -> np.ndarray:
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def most_similar(E: np.ndarray, vocab: Vocab, word: str, k: int = 5) -> list[tuple[str, float]]:
    En = normalize(E)
    sims = En @ En[vocab.index[word]]
    order = [i for i in np.argsort(-sims) if vocab.words[i] != word][:k]
    return [(vocab.words[i], float(sims[i])) for i in order]


def analogy(E: np.ndarray, vocab: Vocab, a: str, b: str, c: str, k: int = 1) -> list[str]:
    """a is to b as c is to ?  ->  the word nearest to (b - a + c), excluding a, b, c (3CosAdd)."""
    En = normalize(E)
    target = En[vocab.index[b]] - En[vocab.index[a]] + En[vocab.index[c]]
    sims = En @ (target / np.linalg.norm(target))
    banned = {a, b, c}
    return [vocab.words[i] for i in np.argsort(-sims) if vocab.words[i] not in banned][:k]


def analogy_accuracy(
    E: np.ndarray, vocab: Vocab, questions: list[tuple[str, str, str, str]]
) -> float:
    return float(np.mean([analogy(E, vocab, a, b, c)[0] == d for a, b, c, d in questions]))


def planted_analogies() -> list[tuple[str, str, str, str]]:
    """Every question "a : b :: c : d" among the 8 nouns that the attributes make well defined:
    for each attribute, either a and b agree (then d keeps c's value), or they differ and c
    has a's value (then d takes b's value). Example: man : king :: woman : queen."""
    by_attrs = {attrs: w for w, attrs in NOUNS.items()}
    questions = []
    for a, A in NOUNS.items():
        for b, B in NOUNS.items():
            if a == b:
                continue
            for c, C in NOUNS.items():
                if c in (a, b):
                    continue
                D = []
                for ai, bi, ci in zip(A, B, C):
                    if ai == bi:
                        D.append(ci)
                    elif ci == ai:
                        D.append(bi)
                    else:
                        break
                else:
                    d = by_attrs[tuple(D)]
                    if d not in (a, b, c):
                        questions.append((a, b, c, d))
    return questions
