from pathlib import Path

import numpy as np
import pytest
import torch

from s5_nlp.bm25.impl import BM25, tokenize
from s5_nlp.corpus import load_corpus, relevant_ids
from s5_nlp.eval_metrics.impl import mean_reciprocal_rank

from .impl import (
    BruteForceIndex,
    IVFIndex,
    info_nce,
    recall_vs_exact,
    reciprocal_rank_fusion,
    train_encoder,
)


@pytest.fixture(scope="module")
def clustered_vectors():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(50, 32))
    X = centres[rng.integers(0, 50, 20_000)] + 0.3 * rng.normal(size=(20_000, 32))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = centres[rng.integers(0, 50, 200)] + 0.3 * rng.normal(size=(200, 32))
    return X, Q / np.linalg.norm(Q, axis=1, keepdims=True)


def test_brute_force_matches_full_sort(clustered_vectors):
    X, Q = clustered_vectors
    idx, sims = BruteForceIndex(X).search(Q[:20], 10)
    expected = np.argsort(-(Q[:20] @ X.T), axis=1)[:, :10]
    np.testing.assert_array_equal(idx, expected)
    assert np.all(np.diff(sims, axis=1) <= 1e-6)


def test_ivf_trades_recall_for_speed(clustered_vectors):
    X, Q = clustered_vectors
    exact, _ = BruteForceIndex(X).search(Q, 10)
    ivf = IVFIndex(X, n_lists=64)
    recalls, work = [], []
    for n_probe in (1, 4, 16, 64):
        approx, _ = ivf.search(Q, 10, n_probe)
        recalls.append(recall_vs_exact(approx, exact))
        work.append(ivf.last_n_compared)
    assert recalls == sorted(recalls) and recalls[-1] == 1.0  # probing every cell is exact
    assert recalls[1] > 0.9 and work[1] < 0.15 * len(X)  # 90%+ recall touching <15% of the data
    assert sum(len(lst) for lst in ivf.lists) == len(X)


def test_info_nce_rewards_matching_pairs():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    assert info_nce(q, q).item() < info_nce(q, q.roll(1, 0)).item()
    assert info_nce(q, q).item() < 0.01  # each query is far more similar to itself


@pytest.mark.slow
def test_trained_encoder_retrieves_held_out_questions():
    passages, queries = load_corpus()
    texts = [p.title + " " + p.text for p in passages]
    relevants = [relevant_ids(passages, q) for q in queries]

    def mrr_of(index_vectors, query_vectors):
        idx, _ = BruteForceIndex(index_vectors).search(query_vectors, 10)
        return mean_reciprocal_rank([[passages[i].pid for i in row] for row in idx], relevants)

    torch.manual_seed(0)
    untrained, _ = train_encoder(passages, steps=0)
    model, history = train_encoder(passages)
    dense = mrr_of(model.encode(texts), model.encode([q.text for q in queries]))
    random_mrr = mrr_of(untrained.encode(texts), untrained.encode([q.text for q in queries]))
    bm25 = BM25().fit([tokenize(t) for t in texts])
    bm25_mrr = mean_reciprocal_rank(
        [[passages[i].pid for i, _ in bm25.search(tokenize(q.text), 10)] for q in queries],
        relevants,
    )
    assert history[-1] < 0.5 * history[0]
    assert dense > random_mrr + 0.05  # training helps...
    assert dense < bm25_mrr  # ...but 400 passages is far too little to beat BM25 (see NOTES.md)
    # Hybrid: fuse the BM25 and dense top-50 rankings.
    q_vecs = model.encode([q.text for q in queries])
    dense_top50, _ = BruteForceIndex(model.encode(texts)).search(q_vecs, 50)
    fused = [
        reciprocal_rank_fusion([[i for i, _ in bm25.search(tokenize(q.text), 50)], list(row)])[:10]
        for q, row in zip(queries, dense_top50)
    ]
    fused_mrr = mean_reciprocal_rank([[passages[i].pid for i in r] for r in fused], relevants)
    (Path(__file__).parent / "results.txt").write_text(
        f"held-out interview questions: {len(queries)}, passages: {len(passages)}\n"
        f"MRR@10 dense (trained) {dense:.3f}, dense (untrained) {random_mrr:.3f}, "
        f"BM25 {bm25_mrr:.3f}, BM25+dense RRF {fused_mrr:.3f}\n"
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.convolve(history, np.ones(20) / 20, mode="valid"))
    ax.set(
        xlabel="step",
        ylabel="InfoNCE loss",
        title=f"dual encoder: MRR {dense:.2f} (BM25 {bm25_mrr:.2f})",
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print(f"dense {dense:.3f} untrained {random_mrr:.3f} bm25 {bm25_mrr:.3f}")
