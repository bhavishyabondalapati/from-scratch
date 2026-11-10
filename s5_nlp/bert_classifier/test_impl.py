import random
from pathlib import Path

import pytest
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from .impl import (
    CLS,
    MASK,
    PAD,
    SEP,
    TinyBERT,
    WordTokenizer,
    accuracy,
    fine_tune,
    make_clause,
    make_dataset,
    mask_tokens,
    pretrain_mlm,
)


def test_labels_follow_negation_and_but():
    rnd = random.Random(0)
    for _ in range(200):
        text, label = make_clause(rnd)
        positive_word = any(
            w in text.split() for w in ["good", "great", "wonderful", "excellent", "fun", "moving"]
        )
        assert label == int(positive_word != ("not" in text.split()))
    texts, labels = make_dataset(500, seed=1)
    for t, y in zip(texts, labels):
        last = t.split(" but ")[-1]
        pos = any(
            w in last.split() for w in ["good", "great", "wonderful", "excellent", "fun", "moving"]
        )
        assert y == int(pos != ("not" in last.split()))
    assert 0.4 < sum(labels) / len(labels) < 0.6


def test_tokenizer_and_masking():
    tok = WordTokenizer(["the movie was good"])
    ids, mask = tok.encode_batch(["the movie was good", "unknown words here"], max_len=8)
    assert ids[0, 0] == CLS and ids[0, 5] == SEP and ids[0, 6] == PAD
    assert mask.sum(1).tolist() == [6, 5]
    big = torch.randint(5, 40, (200, 20))
    corrupted, labels = mask_tokens(big, 40, generator=torch.Generator().manual_seed(0))
    chosen = labels != -100
    assert 0.12 < chosen.float().mean() < 0.18
    assert (corrupted[chosen] == MASK).float().mean() == pytest.approx(0.8, abs=0.05)
    assert torch.equal(corrupted[~chosen], big[~chosen])


def test_encoder_is_bidirectional_and_ignores_padding():
    torch.manual_seed(0)
    model = TinyBERT(30, layers=1, dropout=0.0).eval()
    ids = torch.tensor([[CLS, 7, 8, 9, SEP, PAD, PAD]])
    mask = ids != PAD
    h = model.encode(ids, mask)
    ids2 = ids.clone()
    ids2[0, 3] = 12  # change a LATER word: earlier positions must change too (no causal mask)
    assert not torch.allclose(model.encode(ids2, mask)[0, 1], h[0, 1])
    ids3 = ids.clone()
    ids3[0, 5:] = 13  # change only padding tokens: real positions must not change
    torch.testing.assert_close(model.encode(ids3, mask)[0, :5], h[0, :5])


@pytest.mark.slow
def test_tiny_bert_beats_bag_of_words():
    train_x, train_y = make_dataset(4000, seed=0)
    test_x, test_y = make_dataset(1000, seed=1)
    # Baseline: unigram + bigram counts with logistic regression.
    vec = CountVectorizer(ngram_range=(1, 2), token_pattern=r"\S+")
    bow = LogisticRegression(max_iter=2000).fit(vec.fit_transform(train_x), train_y)
    bow_acc = bow.score(vec.transform(test_x), test_y)

    tok = WordTokenizer(train_x)
    ids, mask = tok.encode_batch(train_x)
    tids, tmask = tok.encode_batch(test_x)
    y, ty = torch.tensor(train_y), torch.tensor(test_y)
    torch.manual_seed(0)
    model = TinyBERT(len(tok))
    mlm_history = pretrain_mlm(model, ids, mask, steps=300)
    history = fine_tune(model, ids, mask, y, epochs=15)
    bert_acc = accuracy(model, tids, tmask, ty)
    assert mlm_history[-1] < 0.7 * mlm_history[0]
    assert bow_acc < 0.9  # order matters: bags of words cannot get "A but not very B" right
    assert bert_acc > 0.97 and bert_acc > bow_acc + 0.05

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].plot(mlm_history)
    axes[0].set(xlabel="step", ylabel="MLM loss", title="masked-LM pre-training")
    axes[1].plot(history)
    axes[1].set(
        xlabel="epoch",
        ylabel="loss",
        yscale="log",
        title=f"fine-tune: test acc {bert_acc:.3f} (BoW {bow_acc:.3f})",
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print(f"bert {bert_acc:.3f} bow {bow_acc:.3f}")
