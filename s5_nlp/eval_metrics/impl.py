"""Text-generation and retrieval metrics: BLEU, ROUGE-L, exact match / F1, recall@k, MRR, NDCG."""

from __future__ import annotations

import math
import re
import string
from collections import Counter

# --- Generation metrics ---------------------------------------------------------------


def ngram_counts(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: list[list[str]],
    references: list[list[list[str]]],
    max_n: int = 4,
    smooth: bool = False,
) -> float:
    """BLEU (Papineni et al. 2002), corpus level, uniform weights over 1..max_n grams.

    p_n = clipped n-gram matches / hypothesis n-grams, summed over the corpus.
    "Clipped": a hypothesis n-gram counts at most as many times as it appears in
    the most generous single reference (so "the the the the" cannot cheat).
    BP = brevity penalty: exp(1 - r/c) if the hypotheses are shorter than the
    closest reference lengths r, else 1 (short outputs would otherwise win on precision).
    BLEU = BP * exp(mean_n log p_n).  smooth=True adds 1 to numerator and denominator
    for n > 1 (Lin & Och's "add-one" smoothing, used for sentence-level scores).
    """
    matches, totals = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for hyp, refs in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += min((len(r) for r in refs), key=lambda L: (abs(L - len(hyp)), L))
        for n in range(1, max_n + 1):
            h = ngram_counts(hyp, n)
            max_ref: Counter = Counter()
            for r in refs:
                max_ref |= ngram_counts(r, n)  # elementwise max over references
            matches[n - 1] += sum(min(c, max_ref[g]) for g, c in h.items())
            totals[n - 1] += max(len(hyp) - n + 1, 0)
    log_p = 0.0
    for n in range(max_n):
        m, t = matches[n], totals[n]
        if smooth and n > 0:
            m, t = m + 1, t + 1
        if m == 0 or t == 0:
            return 0.0
        log_p += math.log(m / t) / max_n
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(log_p)


def lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hypothesis: list[str], reference: list[str], beta: float = 1.0) -> dict[str, float]:
    """ROUGE-L: precision, recall and F based on the longest common subsequence,
    so words must appear in the same order but need not be adjacent."""
    lcs = lcs_length(hypothesis, reference)
    if lcs == 0:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    p, r = lcs / len(hypothesis), lcs / len(reference)
    f = (1 + beta**2) * p * r / (r + beta**2 * p)
    return {"precision": p, "recall": r, "f": f}


def normalize_answer(s: str) -> str:
    """SQuAD normalisation: lowercase, drop punctuation and articles, squeeze spaces."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str | list[str]) -> float:
    golds = [gold] if isinstance(gold, str) else gold
    return float(any(normalize_answer(prediction) == normalize_answer(g) for g in golds))


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD token-overlap F1: partial credit for partially correct answers."""
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return float(p == g)  # both empty counts as a match
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


# --- Retrieval / ranking metrics -----------------------------------------------------------


def recall_at_k(ranked: list, relevant: set, k: int) -> float:
    """Fraction of the relevant items that appear in the top k."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: list, relevant: set) -> float:
    """1 / (rank of the first relevant item), or 0 if none is retrieved."""
    for i, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(rankings: list[list], relevants: list[set]) -> float:
    return sum(reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants)) / len(rankings)


def dcg(gains: list[float], k: int | None = None) -> float:
    """Discounted cumulative gain: sum_i gain_i / log2(i + 1), positions starting at 1.
    Hits near the top count more."""
    gains = gains[:k] if k else gains
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg(
    ranked_gains: list[float], all_gains: list[float] | None = None, k: int | None = None
) -> float:
    """DCG of the ranking divided by the DCG of the ideal ordering (1.0 = perfect).
    ranked_gains: graded relevance of the items in ranked order.
    all_gains: relevance of every candidate (for the ideal); defaults to ranked_gains."""
    ideal = dcg(sorted(all_gains if all_gains is not None else ranked_gains, reverse=True), k)
    return dcg(ranked_gains, k) / ideal if ideal > 0 else 0.0
