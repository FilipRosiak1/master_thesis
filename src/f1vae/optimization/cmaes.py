from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import ObjectiveFn, OptimizationResult

try:
    import cma as _cma
except Exception:
    _cma = None


@dataclass(frozen=True)
class CMAESConfig:
    iterations: int = 80
    population_size: int | None = None
    initial_sigma: float = 0.8
    seed: int | None = None
    use_external_backend: bool = True


def _optimize_with_external_cma(objective: ObjectiveFn, latent_dim: int, cfg: CMAESConfig) -> OptimizationResult:
    if _cma is None:
        raise RuntimeError("External CMA backend not available")

    popsize = cfg.population_size or (4 + int(3 * np.log(latent_dim)))
    options = {
        "popsize": int(popsize),
        "seed": cfg.seed,
        "verbose": -9,
    }
    es = _cma.CMAEvolutionStrategy(np.zeros(latent_dim), cfg.initial_sigma, options)

    best_score = -np.inf
    best_z = np.zeros(latent_dim, dtype=np.float64)
    history: list[float] = []

    for _ in range(cfg.iterations):
        candidates = np.array(es.ask(), dtype=np.float64)
        scores = np.array([float(objective(z)) for z in candidates], dtype=np.float64)
        es.tell(candidates.tolist(), (-scores).tolist())

        idx = int(np.argmax(scores))
        if scores[idx] > best_score:
            best_score = float(scores[idx])
            best_z = candidates[idx].copy()
        history.append(best_score)

    return OptimizationResult(best_z=best_z, best_score=best_score, history_best_score=history)


def _optimize_internal(objective: ObjectiveFn, latent_dim: int, cfg: CMAESConfig) -> OptimizationResult:
    rng = np.random.default_rng(cfg.seed)
    n = latent_dim
    lamb = cfg.population_size or (4 + int(3 * np.log(n)))
    mu = lamb // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)

    cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs

    mean = np.zeros(n, dtype=np.float64)
    sigma = float(cfg.initial_sigma)
    cov = np.eye(n, dtype=np.float64)
    p_c = np.zeros(n, dtype=np.float64)
    p_s = np.zeros(n, dtype=np.float64)

    chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    best_z = mean.copy()
    best_score = -np.inf
    history: list[float] = []

    for gen in range(cfg.iterations):
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        z_samples = rng.standard_normal((lamb, n))
        y_samples = z_samples @ sqrt_cov.T
        x_samples = mean + sigma * y_samples

        scores = np.array([float(objective(x)) for x in x_samples], dtype=np.float64)
        order = np.argsort(scores)[::-1]
        x_selected = x_samples[order[:mu]]

        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best_z = x_samples[order[0]].copy()
        history.append(best_score)

        mean_old = mean.copy()
        mean = np.sum(x_selected * weights[:, None], axis=0)
        y_w = (mean - mean_old) / sigma

        p_s = (1.0 - cs) * p_s + np.sqrt(cs * (2.0 - cs) * mu_eff) * (inv_sqrt_cov @ y_w)
        norm_ps = np.linalg.norm(p_s)
        hsig = float(
            norm_ps
            / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (gen + 1.0)))
            / chi_n
            < (1.4 + 2.0 / (n + 1.0))
        )
        p_c = (1.0 - cc) * p_c + hsig * np.sqrt(cc * (2.0 - cc) * mu_eff) * y_w

        artmp = (x_selected - mean_old) / sigma
        rank_mu = np.zeros_like(cov)
        for i in range(mu):
            rank_mu += weights[i] * np.outer(artmp[i], artmp[i])

        cov = (
            (1.0 - c1 - cmu) * cov
            + c1 * (np.outer(p_c, p_c) + (1.0 - hsig) * cc * (2.0 - cc) * cov)
            + cmu * rank_mu
        )
        cov = 0.5 * (cov + cov.T)
        sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1.0)))

    return OptimizationResult(best_z=best_z, best_score=best_score, history_best_score=history)


def optimize_latent_cmaes(
    objective: ObjectiveFn,
    latent_dim: int,
    *,
    config: CMAESConfig | None = None,
) -> OptimizationResult:
    cfg = config or CMAESConfig()
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    if cfg.initial_sigma <= 0.0:
        raise ValueError("initial_sigma must be positive")

    if cfg.use_external_backend and _cma is not None:
        return _optimize_with_external_cma(objective, latent_dim, cfg)
    return _optimize_internal(objective, latent_dim, cfg)
