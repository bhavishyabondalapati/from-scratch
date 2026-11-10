from pathlib import Path

import pytest
import torch
from sklearn.neighbors import KNeighborsClassifier
from torch.distributions import Normal, kl_divergence

from s4_deep_learning.torch_utils import digits_tensors, save_curves

from .impl import VAE, importance_sampled_nll, kl_to_standard_normal, negative_elbo, train


@pytest.fixture(scope="module")
def data():
    X, y, Xte, yte = digits_tensors()
    return X.reshape(len(X), -1), y, Xte.reshape(len(Xte), -1), yte


@pytest.fixture(scope="module")
def trained(data):
    X, *_ = data
    torch.manual_seed(0)
    model = VAE(latent=4)
    return model, train(model, X, epochs=300)


def test_kl_closed_form_matches_torch_distributions():
    mu, logvar = torch.randn(5, 3), torch.randn(5, 3)
    ref = kl_divergence(Normal(mu, torch.exp(0.5 * logvar)), Normal(0.0, 1.0)).sum(1)
    torch.testing.assert_close(kl_to_standard_normal(mu, logvar), ref)
    assert torch.all(kl_to_standard_normal(torch.zeros(2, 3), torch.zeros(2, 3)) == 0)


def test_reparameterisation_lets_gradients_reach_mu_and_sigma():
    mu = torch.zeros(1000, 1, requires_grad=True)
    logvar = torch.zeros(1000, 1, requires_grad=True)
    z = VAE.reparameterize(mu, logvar)
    (
        z**2
    ).mean().backward()  # E[z^2] = mu^2 + sigma^2  =>  d/dlogvar = sigma^2 = 1 (per element avg)
    assert mu.grad is not None and logvar.grad is not None
    assert logvar.grad.sum().item() == pytest.approx(1.0, abs=0.15)


def test_elbo_bounds_the_likelihood_and_beats_pixel_baseline(data, trained):
    X, _, Xte, _ = data
    model, history = trained
    torch.manual_seed(1)
    with torch.no_grad():
        elbo_nll = negative_elbo(model, Xte)[0].item()
    iw_nll = importance_sampled_nll(model, Xte, k=200)
    assert iw_nll <= elbo_nll + 0.5  # the ELBO is a lower bound on log p(x)
    # Baseline: every pixel an independent Bernoulli with the training-set mean.
    p = X.mean(0).clamp(1e-3, 1 - 1e-3)
    baseline = float(-(Xte * p.log() + (1 - Xte) * (1 - p).log()).sum(1).mean())
    assert elbo_nll < 0.9 * baseline  # ~12% better; see NOTES.md for why the margin is modest
    assert history["loss"][-1] < history["loss"][0]
    curves = {k: v for k, v in history.items()}
    save_curves(
        Path(__file__).parent / "training_curve.png",
        curves,
        f"VAE (latent 4): test -ELBO {elbo_nll:.1f}, IW-NLL {iw_nll:.1f} nats",
        ylabel="nats / image",
    )


def test_latent_space_organises_digits(data, trained):
    X, y, Xte, yte = data
    model, _ = trained
    with torch.no_grad():
        Z, Zte = model.encode(X)[0].numpy(), model.encode(Xte)[0].numpy()
    assert (
        KNeighborsClassifier(5).fit(Z, y).score(Zte, yte) > 0.8
    )  # 4 numbers per image, no labels used


def test_samples_from_the_prior_look_like_digits(data, trained):
    X, y, _, _ = data
    model, _ = trained
    torch.manual_seed(2)
    samples = model.sample(200)
    noise = torch.rand(200, 64)
    clf = KNeighborsClassifier(1).fit(X.numpy(), y.numpy())
    d_samples = clf.kneighbors(samples.numpy())[0].mean()
    d_noise = clf.kneighbors(noise.numpy())[0].mean()
    assert d_samples < 0.5 * d_noise  # generated images lie close to real digits
