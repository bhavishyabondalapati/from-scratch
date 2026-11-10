import math
from pathlib import Path

import pytest
import torch

from .impl import (
    GAN,
    discriminator_loss,
    generator_loss,
    mode_metrics,
    ring_centres,
    ring_of_gaussians,
)


def test_data_and_metrics():
    real = ring_of_gaussians(4000, generator=torch.Generator().manual_seed(0))
    modes, quality = mode_metrics(real)
    assert modes == 8 and quality > 0.98  # the real data covers every mode
    one_mode = ring_centres()[:1].repeat(500, 1)
    assert mode_metrics(one_mode)[0] == 1  # collapse onto a single mode is detected


def test_losses_match_their_formulas():
    d_real, d_fake = torch.randn(10), torch.randn(10)
    s = torch.sigmoid
    expected_d = -(torch.log(s(d_real)).mean() + torch.log(1 - s(d_fake)).mean())
    torch.testing.assert_close(discriminator_loss(d_real, d_fake), expected_d)
    torch.testing.assert_close(generator_loss(d_fake), -torch.log(s(d_fake)).mean())
    # When D cannot tell real from fake (D = 1/2 everywhere), its loss is 2 log 2.
    z = torch.zeros(5)
    assert discriminator_loss(z, z).item() == pytest.approx(2 * math.log(2))


def test_discriminator_updates_improve_discrimination():
    """With the generator frozen, D's updates alone must lower D's loss."""
    gan = GAN(seed=0)
    real = ring_of_gaussians(512, generator=torch.Generator().manual_seed(1))
    fake = gan.sample(512)
    with torch.no_grad():
        before = discriminator_loss(gan.D(real), gan.D(fake)).item()
    for _ in range(50):
        loss = discriminator_loss(gan.D(real), gan.D(fake))
        gan.opt_d.zero_grad()
        loss.backward()
        gan.opt_d.step()
    with torch.no_grad():
        after = discriminator_loss(gan.D(real), gan.D(fake)).item()
    assert after < 0.5 * before


@pytest.mark.slow
def test_gan_learns_most_modes_of_the_ring():
    gan = GAN(seed=0)
    untrained_modes, untrained_quality = mode_metrics(gan.sample(2000))
    history = gan.train(steps=3000)
    modes, quality = mode_metrics(gan.sample(4000))
    assert untrained_modes <= 1 and untrained_quality < 0.05
    assert modes >= 6 and quality > 0.5

    import matplotlib.pyplot as plt

    samples, real = gan.sample(2000), ring_of_gaussians(2000)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["d_loss"], lw=0.5, label="D loss")
    axes[0].plot(history["g_loss"], lw=0.5, label="G loss")
    axes[0].axhline(2 * math.log(2), ls="--", c="gray", label="2 log 2")
    axes[0].set(xlabel="step", title=f"modes covered {modes}/8, quality {quality:.2f}")
    axes[0].legend(fontsize=8)
    axes[1].scatter(*real.T, s=2, alpha=0.3, label="real")
    axes[1].scatter(*samples.T, s=2, alpha=0.3, label="generated")
    axes[1].set(aspect="equal", title="samples")
    axes[1].legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "training_curve.png", dpi=110)
    plt.close(fig)
    print(f"modes {modes} quality {quality:.3f}")
