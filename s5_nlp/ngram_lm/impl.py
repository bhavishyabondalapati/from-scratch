"""n-gram language models with add-k, interpolated, and Kneser-Ney smoothing; perplexity."""

from __future__ import annotations

import math
from collections import Counter

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


def build_vocab(sentences: list[list[str]], min_count: int = 1) -> set[str]:
    counts = Counter(w for s in sentences for w in s)
    return {w for w, c in counts.items() if c >= min_count} | {EOS, UNK}


def prepare(sentence: list[str], n: int, vocab: set[str]) -> list[str]:
    """Map rare/unseen words to <unk>, pad with n-1 start symbols and one end symbol."""
    return [BOS] * (n - 1) + [w if w in vocab else UNK for w in sentence] + [EOS]


class NGramLM:
    """P(w | previous n-1 words), estimated from counts.

    smoothing:
      "mle"         count(h, w) / count(h): zero for anything unseen -> infinite perplexity
      "add_k"       (count(h, w) + k) / (count(h) + k V): pretend every word was seen k more times
      "interpolate" lambda-weighted mix of the n-gram, (n-1)-gram, ..., unigram MLEs
      "kneser_ney"  interpolated Kneser-Ney with absolute discount d (the classic strong baseline)
    """

    def __init__(
        self,
        n: int = 2,
        smoothing: str = "kneser_ney",
        k: float = 1.0,
        discount: float = 0.75,
        lambdas: list[float] | None = None,
    ) -> None:
        if smoothing not in {"mle", "add_k", "interpolate", "kneser_ney"}:
            raise ValueError(f"unknown smoothing {smoothing!r}")
        self.n, self.smoothing, self.k, self.d = n, smoothing, k, discount
        self.lambdas = lambdas or [1.0 / n] * n  # weights for orders n, n-1, ..., 1

    def fit(self, sentences: list[list[str]], vocab: set[str] | None = None) -> NGramLM:
        self.vocab = vocab or build_vocab(sentences)
        self.V = len(self.vocab)  # <s> is never predicted, so it is not in the vocab
        # counts[m][(h, w)] for every order m = 1..n, where h has m-1 words.
        self.counts = [Counter() for _ in range(self.n + 1)]
        self.context_counts = [Counter() for _ in range(self.n + 1)]
        for s in sentences:
            toks = prepare(s, self.n, self.vocab)
            for i in range(self.n - 1, len(toks)):
                for m in range(1, self.n + 1):
                    h, w = tuple(toks[i - m + 1 : i]), toks[i]
                    self.counts[m][(h, w)] += 1
                    self.context_counts[m][h] += 1
        if self.smoothing == "kneser_ney":
            self._prepare_kneser_ney()
        return self

    # --- Kneser-Ney ----------------------------------------------------------------------

    def _prepare_kneser_ney(self) -> None:
        """Lower orders use CONTINUATION counts: not "how often does w occur" but
        "in how many different contexts does w occur". ("Francisco" is frequent but
        almost only after "San", so it should be an unlikely guess after a new word.)

        For every order m we store, for the distribution actually used at that order
        (raw counts at the top order, continuation counts below it):
          num[m][(h, w)]  numerator counts
          den[m][h]       their total for context h
          types[m][h]     number of distinct w seen after h (sets the back-off weight)
        """
        self.num = [Counter() for _ in range(self.n + 1)]
        self.num[self.n] = self.counts[self.n]
        for m in range(2, self.n + 1):
            for h, w in self.counts[m]:
                self.num[m - 1][(h[1:], w)] += 1  # one more distinct left context for (h[1:], w)
        self.den = [Counter() for _ in range(self.n + 1)]
        self.types = [Counter() for _ in range(self.n + 1)]
        for m in range(1, self.n + 1):
            for (h, _w), c in self.num[m].items():
                self.den[m][h] += c
                self.types[m][h] += 1

    def _kn(self, h: tuple[str, ...], w: str, m: int) -> float:
        """P_KN(w | h) = max(c(h, w) - d, 0) / c(h) + d * types(h) / c(h) * P_KN(w | h[1:])."""
        if m == 0:
            return 1.0 / self.V  # uniform base distribution
        lower = self._kn(h[1:], w, m - 1)
        total = self.den[m][h]
        if total == 0:
            return lower  # unseen context: back off completely
        discounted = max(self.num[m][(h, w)] - self.d, 0) / total
        return discounted + self.d * self.types[m][h] / total * lower

    # --- probabilities ------------------------------------------------------------------------

    def _mle(self, h: tuple[str, ...], w: str, m: int) -> float:
        c = self.context_counts[m][h]
        return self.counts[m][(h, w)] / c if c else 0.0

    def prob(self, word: str, history: tuple[str, ...]) -> float:
        w = word if word in self.vocab else UNK
        h = tuple(history[-(self.n - 1) :]) if self.n > 1 else ()
        if self.smoothing == "mle":
            return self._mle(h, w, self.n)
        if self.smoothing == "add_k":
            return (self.counts[self.n][(h, w)] + self.k) / (
                self.context_counts[self.n][h] + self.k * self.V
            )
        if self.smoothing == "interpolate":
            # Higher orders are plain MLEs; the unigram is add-1 smoothed so that every
            # word keeps some probability. If a context was never seen, its weight is
            # passed down to the next lower order, so the mixture always sums to one.
            p, carry = 0.0, 0.0
            for lam, m in zip(self.lambdas, range(self.n, 0, -1)):
                lam += carry
                if m == 1:
                    total = self.context_counts[1][()]
                    p += lam * (self.counts[1][((), w)] + 1) / (total + self.V)
                elif self.context_counts[m][h[len(h) - (m - 1) :]] == 0:
                    carry = lam
                else:
                    carry = 0.0
                    p += lam * self._mle(h[len(h) - (m - 1) :], w, m)
            return p
        return self._kn(h, w, self.n)

    def sentence_logprob(self, sentence: list[str]) -> tuple[float, int]:
        """(sum of natural-log probabilities, number of predicted tokens incl. </s>)."""
        toks = prepare(sentence, self.n, self.vocab)
        total = 0.0
        for i in range(self.n - 1, len(toks)):
            p = self.prob(toks[i], tuple(toks[i - self.n + 1 : i]))
            if p == 0:
                return -math.inf, len(toks) - self.n + 1
            total += math.log(p)
        return total, len(toks) - self.n + 1

    def perplexity(self, sentences: list[list[str]]) -> float:
        """exp(average negative log-probability per token). Lower is better;
        the uniform model over V words has perplexity exactly V."""
        logp, count = 0.0, 0
        for s in sentences:
            lp, c = self.sentence_logprob(s)
            logp += lp
            count += c
        return math.exp(-logp / count) if logp > -math.inf else math.inf


def tune_interpolation(
    train: list[list[str]], dev: list[list[str]], n: int, grid: int = 5
) -> list[float]:
    """Pick interpolation weights on held-out data (a tiny grid search)."""
    best, best_ppl = None, math.inf
    steps = [i / grid for i in range(1, grid)]
    candidates = (
        [[a, 1 - a] for a in steps]
        if n == 2
        else [[a, b, 1 - a - b] for a in steps for b in steps if a + b < 1]
    )
    model = NGramLM(n, "interpolate").fit(train)
    for lam in candidates:
        model.lambdas = lam
        ppl = model.perplexity(dev)
        if ppl < best_ppl:
            best, best_ppl = lam, ppl
    return best


def next_word_distribution(model: NGramLM, history: tuple[str, ...]) -> dict[str, float]:
    return {w: model.prob(w, history) for w in model.vocab}


def generate(model: NGramLM, max_len: int = 20, seed: int = 0) -> list[str]:
    import random

    rnd = random.Random(seed)
    history = [BOS] * (model.n - 1)
    out: list[str] = []
    words = sorted(model.vocab - {UNK})
    for _ in range(max_len):
        weights = [model.prob(w, tuple(history)) for w in words]
        w = rnd.choices(words, weights)[0]
        if w == EOS:
            break
        out.append(w)
        history.append(w)
    return out
