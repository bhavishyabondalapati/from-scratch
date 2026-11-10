"""Shared pytest setup: headless matplotlib and a seeded NumPy generator."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh, seeded random generator so every test is reproducible."""
    return np.random.default_rng(0)
