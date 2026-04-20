from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1vae.optimization import CEMConfig, CMAESConfig, optimize_latent_cem, optimize_latent_cmaes


def _objective(z: np.ndarray) -> float:
    target = np.array([0.5, -1.0, 2.0], dtype=np.float64)
    return -float(np.sum((z - target) ** 2))


def test_cmaes_optimizer_improves_score():
    result = optimize_latent_cmaes(
        _objective,
        latent_dim=3,
        config=CMAESConfig(iterations=40, population_size=20, initial_sigma=0.8, seed=7),
    )
    assert result.best_score > -0.5


def test_cem_optimizer_improves_score():
    result = optimize_latent_cem(
        _objective,
        latent_dim=3,
        config=CEMConfig(iterations=40, population_size=80, elite_fraction=0.2, seed=7),
    )
    assert result.best_score > -0.5
