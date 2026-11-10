from pathlib import Path

import pytest
import torch

from s4_deep_learning.torch_utils import digits_tensors, save_curves

from .impl import BasicBlock, SmallResNet, accuracy, first_layer_grad_norm, train


def test_block_shapes_and_shortcut():
    x = torch.randn(2, 16, 8, 8)
    assert BasicBlock(16, 16)(x).shape == (2, 16, 8, 8)
    down = BasicBlock(16, 32, stride=2)
    assert down(x).shape == (2, 32, 4, 4)
    assert isinstance(down.shortcut, torch.nn.Sequential)
    assert isinstance(BasicBlock(16, 16).shortcut, torch.nn.Identity)


def test_block_is_identity_plus_residual():
    torch.manual_seed(0)
    block = BasicBlock(4, 4).eval()
    torch.nn.init.zeros_(block.bn2.weight)  # "zero-init" the last BN: F(x) == 0
    x = torch.relu(torch.randn(3, 4, 5, 5))  # non-negative input, so the final ReLU is a no-op
    torch.testing.assert_close(block(x), x)


def test_residual_connections_keep_gradients_balanced_across_depth():
    """In a 30-block plain net (with batch norm) the gradient at the first block is
    thousands of times larger than at the last one: it explodes going backwards.
    With skip connections every block sees a gradient of similar size."""
    X, y, _, _ = digits_tensors()

    def spread(residual):
        torch.manual_seed(1)
        model = SmallResNet(30, residual=residual)
        first_layer_grad_norm(model, X[:64], y[:64])  # runs one backward pass
        norms = [b.conv1.weight.grad.norm().item() for b in model.blocks]
        return norms[0] / norms[-1]

    assert spread(residual=False) > 1e3
    assert spread(residual=True) < 20


@pytest.mark.slow
def test_deep_resnet_trains_better_than_deep_plain_net():
    X, y, Xte, yte = digits_tensors()
    curves, accs = {}, {}
    for name, residual in [("plain (20 blocks)", False), ("residual (20 blocks)", True)]:
        torch.manual_seed(0)
        model = SmallResNet(20, residual=residual)
        curves[name] = train(model, X, y, epochs=8, lr=0.05)
        accs[name] = accuracy(model, Xte, yte)
    assert accs["residual (20 blocks)"] > 0.95
    assert curves["residual (20 blocks)"][-1] < curves["plain (20 blocks)"][-1]
    labelled = {f"{k} test acc {accs[k]:.3f}": v for k, v in curves.items()}
    save_curves(
        Path(__file__).parent / "training_curve.png", labelled, "42-conv nets on digits", logy=True
    )
