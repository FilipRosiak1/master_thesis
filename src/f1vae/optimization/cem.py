from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ObjectiveFn, OptimizationResult


@dataclass(frozen=True)
class CEMConfig:
    iterations: int = 80
    population_size: int = 96
    elite_fraction: float = 0.2
    initial_std: float = 1.0
    smoothing: float = 0.2
    min_std: float = 1e-3
    seed: int | None = None


def optimize_latent_cem(
    objective: ObjectiveFn,
    latent_dim: int,
    *,
    config: CEMConfig | None = None,
) -> OptimizationResult:
    cfg = config or CEMConfig()
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if cfg.population_size < 2:
        raise ValueError("population_size must be >= 2")
    if not (0.0 < cfg.elite_fraction <= 0.5):
        raise ValueError("elite_fraction must be in (0.0, 0.5]")

    rng = np.random.default_rng(cfg.seed)
    mean = np.zeros(latent_dim, dtype=np.float64)
    std = np.full(latent_dim, cfg.initial_std, dtype=np.float64)

    elite_count = max(2, int(cfg.population_size * cfg.elite_fraction))
    best_z = mean.copy()
    best_score = -np.inf
    history: list[float] = []

    for _ in range(cfg.iterations):
        population = rng.normal(loc=mean, scale=std, size=(cfg.population_size, latent_dim))
        scores = np.array([float(objective(z)) for z in population], dtype=np.float64)
        order = np.argsort(scores)[::-1]
        elite = population[order[:elite_count]]

        elite_mean = elite.mean(axis=0)
        elite_std = elite.std(axis=0)
        mean = cfg.smoothing * mean + (1.0 - cfg.smoothing) * elite_mean
        std = cfg.smoothing * std + (1.0 - cfg.smoothing) * elite_std
        std = np.maximum(std, cfg.min_std)

        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best_z = population[order[0]].copy()
        history.append(best_score)

    return OptimizationResult(best_z=best_z, best_score=best_score, history_best_score=history)
