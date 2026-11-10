from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from s4_deep_learning.torch_utils import digits_tensors

from .impl import EmbeddingNet, contrastive_loss, make_pairs, pair_distances, train


def test_contrastive_loss_formula():
    e1 = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    e2 = torch.tensor([[0.3, 0.4], [0.3, 0.4], [3.0, 4.0]])  # distances 0.5, 0.5, 5.0
    same = torch.tensor([True, False, False])
    # same at d=.5 -> .25; different at d=.5 -> (1-.5)^2=.25; different at d=5 -> 0 (beyond margin)
    expected = (0.25 + 0.25 + 0.0) / 3 / 2
    assert contrastive_loss(e1, e2, same).item() == pytest.approx(expected, abs=1e-6)


def test_pairs_are_balanced_and_labelled_correctly():
    X, y, _, _ = digits_tensors()
    gen = torch.Generator().manual_seed(0)
    a, b, same = make_pairs(X, y, 200, gen)
    assert a.shape == b.shape == (200, 1, 8, 8) and same.float().mean() == 0.5
    # Recover labels by exact image matching and check the "same" flag.
    flat = X.reshape(len(X), -1)
    label_of = lambda img: int(y[(flat == img.reshape(1, -1)).all(1)][0])  # noqa: E731
    assert all((label_of(a[k]) == label_of(b[k])) == bool(same[k]) for k in range(20))


def test_weights_are_shared():
    model = EmbeddingNet()
    x = torch.rand(2, 1, 8, 8)
    e = model(torch.cat([x, x]))
    torch.testing.assert_close(e[:2], e[2:])  # same image, same embedding, either side of the pair


@pytest.mark.slow
def test_verification_on_new_images_and_on_unseen_classes():
    X, y, Xte, yte = digits_tensors()
    seen = y < 7
    torch.manual_seed(0)
    model = EmbeddingNet()
    history = train(model, X[seen], y[seen], steps=800)

    def aucs(images, labels, seed):
        a, b, same = make_pairs(images, labels, 2000, torch.Generator().manual_seed(seed))
        learned = roc_auc_score(same.numpy(), -pair_distances(model, a, b).numpy())
        pixels = roc_auc_score(same.numpy(), -torch.norm((a - b).flatten(1), dim=1).numpy())
        return learned, pixels

    # 1) New test images of the 7 classes seen in training: near-perfect verification.
    seen_learned, seen_pixels = aucs(Xte[yte < 7], yte[yte < 7], seed=1)
    assert seen_learned > 0.99 > seen_pixels + 0.1
    # 2) Classes 7, 8, 9 never seen in training. Honest finding (see NOTES.md): with only 7
    #    training classes the embedding specialises and transfers WORSE than raw pixels,
    #    though still clearly better than chance.
    unseen_learned, unseen_pixels = aucs(Xte[yte >= 7], yte[yte >= 7], seed=2)
    assert 0.6 < unseen_learned

    import matplotlib.pyplot as plt

    smooth = np.convolve(history, np.ones(25) / 25, mode="valid")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(smooth)
    ax.set(
        xlabel="step",
        ylabel="contrastive loss (smoothed)",
        title=f"verification AUC: seen classes {seen_learned:.3f}, unseen {unseen_learned:.3f}",
    )
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print(
        f"seen {seen_learned:.3f}/{seen_pixels:.3f} unseen {unseen_learned:.3f}/{unseen_pixels:.3f}"
    )
