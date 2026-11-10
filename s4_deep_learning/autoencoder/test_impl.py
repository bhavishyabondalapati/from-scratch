from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from s4_deep_learning.torch_utils import digits_tensors, save_curves

from .impl import Autoencoder, reconstruction_mse, train


@pytest.fixture(scope="module")
def data():
    X, y, Xte, yte = digits_tensors()
    return X.reshape(len(X), -1), y, Xte.reshape(len(Xte), -1), yte


@pytest.fixture(scope="module")
def trained(data):
    X, _, Xte, _ = data
    torch.manual_seed(0)
    model = Autoencoder(code_dim=8)
    history = train(model, X, epochs=200)
    return model, history


def test_shapes():
    out = Autoencoder(4)(torch.rand(5, 64))
    assert out.shape == (5, 64) and out.min() >= 0 and out.max() <= 1


def test_linear_autoencoder_matches_pca(data):
    X, _, Xte, _ = data
    k = 8
    pca = PCA(k).fit(X.numpy())
    pca_err = float(np.mean((pca.inverse_transform(pca.transform(Xte.numpy())) - Xte.numpy()) ** 2))
    torch.manual_seed(0)
    lin = Autoencoder(k, linear=True)
    train(lin, X, epochs=400, lr=1e-2)
    assert reconstruction_mse(lin, Xte) == pytest.approx(pca_err, rel=0.05)  # PCA is optimal here


def test_nonlinear_autoencoder_beats_pca_and_mean_image(data, trained):
    X, _, Xte, _ = data
    model, history = trained
    pca = PCA(8).fit(X.numpy())
    pca_err = float(np.mean((pca.inverse_transform(pca.transform(Xte.numpy())) - Xte.numpy()) ** 2))
    mean_err = float(((Xte - X.mean(0)) ** 2).mean())
    ae_err = reconstruction_mse(model, Xte)
    assert ae_err < pca_err < mean_err
    assert history[-1] < 0.2 * history[0]
    save_curves(
        Path(__file__).parent / "training_curve.png",
        {f"train (test MSE {ae_err:.4f}, PCA-8 {pca_err:.4f})": history},
        "Autoencoder 64-128-8-128-64 on digits",
        ylabel="MSE",
        logy=True,
    )


def test_codes_are_useful_features(data, trained):
    X, y, Xte, yte = data
    model, _ = trained
    with torch.no_grad():
        Z, Zte = model.encoder(X).numpy(), model.encoder(Xte).numpy()
    acc = KNeighborsClassifier(5).fit(Z, y).score(Zte, yte)  # labels were never used in training
    assert acc > 0.9


def test_denoising(data):
    X, _, Xte, _ = data
    torch.manual_seed(1)
    model = Autoencoder(16)
    train(model, X, epochs=150, noise=0.3)
    noisy = Xte + 0.3 * torch.randn_like(Xte)
    with torch.no_grad():
        cleaned = model(noisy)
    assert float(((cleaned - Xte) ** 2).mean()) < 0.3 * float(((noisy - Xte) ** 2).mean())
