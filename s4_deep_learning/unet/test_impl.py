from pathlib import Path

import pytest
import torch

from s4_deep_learning.torch_utils import get_device, save_curves

from .impl import UNet, dice_loss, iou, make_shapes, predict, train


def test_dataset_and_shapes():
    X, Y = make_shapes(8, seed=0)
    assert X.shape == Y.shape == (8, 1, 32, 32)
    assert set(Y.unique().tolist()) <= {0.0, 1.0}
    assert UNet()(X).shape == (8, 1, 32, 32)
    assert UNet(skips=False)(X).shape == (8, 1, 32, 32)


def test_metrics():
    a = torch.zeros(1, 1, 4, 4)
    a[..., :2, :] = 1  # top half
    b = torch.zeros(1, 1, 4, 4)
    b[..., :, :2] = 1  # left half
    assert iou(a, a) == 1.0 and iou(a, b) == pytest.approx(4 / 12)
    big = torch.full((1, 1, 4, 4), 20.0)
    assert dice_loss(big, a).item() == pytest.approx(1 - (2 * 8 + 1) / (16 + 8 + 1), abs=1e-6)
    assert dice_loss(torch.where(a > 0, 20.0, -20.0), a).item() < 1e-6


@pytest.mark.slow
def test_unet_segments_discs_and_skips_help():
    Xtr, Ytr = make_shapes(800, seed=0)
    Xte, Yte = make_shapes(200, seed=1)
    # Baseline: threshold the brightness. It catches discs AND squares.
    threshold_iou = iou((Xte > 0.5).float(), Yte)
    torch.manual_seed(0)
    unet = UNet(skips=True)
    hist = train(unet, Xtr, Ytr, device=get_device())
    torch.manual_seed(0)
    no_skip = UNet(skips=False)
    hist_ns = train(no_skip, Xtr, Ytr, device=get_device())
    unet_iou, ns_iou = iou(predict(unet, Xte), Yte), iou(predict(no_skip, Xte), Yte)
    assert threshold_iou < 0.6
    assert unet_iou > 0.85
    assert unet_iou > ns_iou  # skips restore boundary detail lost by pooling
    save_curves(
        Path(__file__).parent / "training_curve.png",
        {f"U-Net (test IoU {unet_iou:.3f})": hist, f"no skips (IoU {ns_iou:.3f})": hist_ns},
        f"Disc segmentation; brightness threshold IoU {threshold_iou:.2f}",
        ylabel="BCE + Dice loss",
        logy=True,
    )
    print(f"threshold {threshold_iou:.3f} unet {unet_iou:.3f} no-skip {ns_iou:.3f}")
