from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1vae.config.defaults import DEFAULTS
from f1vae.config.io import maybe_resolve_path
from f1vae.data.datasets import GrammarRuleDataset
from f1vae.fitness.framsticks import FramsticksFitness, INVALID_FITNESS
from f1vae.inference.checkpoints import load_checkpoint
from f1vae.inference.decode import decode_grammar_indices
from f1vae.inference.encode import encode_grammar_rule_string
from f1vae.models.registry import build_model, dataset_class_for_model


TREE_MODELS = {
    "tree_vae",
    "tree_vae_masked",
    "tree_vae_masked_lhs",
    "tree_vae_masked_lhs_depth",
    "tree_vae_masked_lhs_cond",
}
MODIFIERS = set("RrQqCcLlWwMmIiFfAaSsEe")
DEFAULT_LABELS = "02_lhs_latent256_seed42_best,09_tree_vae_epoch80,10_transformer_vae_epoch90"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seeded latent optimization for selected checkpoints")
    parser.add_argument("--checkpoints-root", default="exports/f1_selected_10_ckpts_20260520_184738")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated checkpoint folder labels")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--framsticks-path", default="src/framsticks/Framsticks54")
    parser.add_argument("--framsticks-lib", default=None)
    parser.add_argument("--framsticks-sim", default="src/framsticks/framspy/eval-allcriteria.sim")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed-candidates", type=int, default=5)
    parser.add_argument("--seed-source", choices=("dataset_top", "simplest", "framsticks_mutations"), default="dataset_top")
    parser.add_argument("--seed-min-fitness", type=float, default=None)
    parser.add_argument("--seed-max-fitness", type=float, default=None)
    parser.add_argument("--algorithm", choices=("cem", "cmaes"), default="cem")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--time-budget-seconds", type=float, default=None, help="Optional time budget per label")
    parser.add_argument("--split-time-budget-across-seeds", action="store_true")
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--initial-std", type=float, default=0.35)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--min-std", type=float, default=0.02)
    parser.add_argument("--mutation-pool-size", type=int, default=200)
    parser.add_argument("--mutation-attempts", type=int, default=2000)
    parser.add_argument("--cma-sigma", type=float, default=None)
    parser.add_argument("--condition-fitness", type=float, default=None, help="Raw target fitness for conditional decoders")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output", default=None)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_checkpoint(run_dir: Path) -> Path:
    best = run_dir / "best_val_recon.pth"
    if best.exists():
        return best
    checkpoints = sorted(run_dir.rglob("*.pth"), key=lambda path: path.as_posix())
    if not checkpoints:
        raise FileNotFoundError(f"No .pth checkpoint found under {run_dir}")
    return checkpoints[0]


def _model_config(run_config: dict[str, Any], checkpoint_meta: dict[str, Any]) -> dict[str, Any]:
    model_name = run_config.get("model") or checkpoint_meta.get("model_name")
    if model_name is None:
        raise ValueError("Missing model name in run_config.json and checkpoint metadata")
    defaults = DEFAULTS[model_name]
    return {
        "model_name": model_name,
        "latent_dim": int(run_config.get("latent_dim") or checkpoint_meta.get("latent_dim") or defaults.latent_dim),
        "hidden_dim": run_config.get("hidden_dim") or checkpoint_meta.get("hidden_dim") or defaults.hidden_dim,
        "embedding_dim": run_config.get("embedding_dim") or checkpoint_meta.get("embedding_dim") or defaults.embedding_dim,
        "max_length": int(run_config.get("max_length") or checkpoint_meta.get("max_length") or defaults.max_length),
    }


def _parse_element(genotype: str, pos: int, stops: set[str]) -> int | None:
    while pos < len(genotype) and genotype[pos] in MODIFIERS:
        pos += 1
    if pos >= len(genotype) or genotype[pos] in stops:
        return None
    if genotype[pos] == "X":
        return pos + 1
    if genotype[pos] != "(":
        return None
    pos += 1
    while pos < len(genotype) and genotype[pos] != ")":
        if genotype[pos] == ",":
            pos += 1
            continue
        pos = _parse_sequence(genotype, pos, {",", ")"})
        if pos is None:
            return None
        if pos < len(genotype) and genotype[pos] == ",":
            pos += 1
    if pos >= len(genotype) or genotype[pos] != ")":
        return None
    return pos + 1


def _parse_sequence(genotype: str, pos: int, stops: set[str]) -> int | None:
    first = _parse_element(genotype, pos, stops)
    if first is None:
        return None
    pos = first
    while pos < len(genotype) and genotype[pos] not in stops:
        pos = _parse_element(genotype, pos, stops)
        if pos is None:
            return None
    return pos


def _is_valid_f1(genotype: str) -> bool:
    genotype = "".join(genotype.split())
    if not genotype:
        return False
    end = _parse_sequence(genotype, 0, set())
    return end == len(genotype)


def _condition_tensor(value: float | None, batch_size: int, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return torch.full((batch_size,), float(value), dtype=torch.float32, device=device)


def _decode_from_z(model_name: str, model, z: np.ndarray, device: torch.device, condition_value: float | None) -> str:
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if model_name in TREE_MODELS:
            condition = _condition_tensor(condition_value, z_tensor.size(0), device)
            if condition is None or not hasattr(model, "condition_dim"):
                logits = model.decode(z_tensor, None, teacher_forcing_ratio=0.0).squeeze(0)
            else:
                logits = model.decode(z_tensor, None, teacher_forcing_ratio=0.0, fitness_condition=condition).squeeze(0)
        else:
            logits = model.decoder(z_tensor, None, teacher_forcing_ratio=0.0).squeeze(0)
        return decode_grammar_indices(logits.argmax(dim=-1))


def _decode_many(model_name: str, model, z_values: np.ndarray, device: torch.device, condition_value: float | None) -> list[str]:
    z_tensor = torch.tensor(z_values, dtype=torch.float32, device=device)
    with torch.no_grad():
        if model_name in TREE_MODELS:
            condition = _condition_tensor(condition_value, z_tensor.size(0), device)
            if condition is None or not hasattr(model, "condition_dim"):
                logits = model.decode(z_tensor, None, teacher_forcing_ratio=0.0)
            else:
                logits = model.decode(z_tensor, None, teacher_forcing_ratio=0.0, fitness_condition=condition)
        else:
            logits = model.decoder(z_tensor, None, teacher_forcing_ratio=0.0)
    tokens = logits.argmax(dim=-1)
    genotypes: list[str] = []
    for row in range(tokens.size(0)):
        try:
            genotypes.append(decode_grammar_indices(tokens[row]))
        except Exception:
            genotypes.append("")
    return genotypes


def _encode_mu(model_name: str, model, batch: torch.Tensor) -> torch.Tensor:
    if model_name in TREE_MODELS:
        mu, _ = model.encode(batch)
        return mu
    mu, _ = model.encoder(batch)
    return mu


def _fitness_in_range(value: float, min_fitness: float | None, max_fitness: float | None) -> bool:
    if math.isnan(value) or math.isinf(value):
        return False
    if min_fitness is not None and value < min_fitness:
        return False
    if max_fitness is not None and value > max_fitness:
        return False
    return True


def _top_dataset_indices(
    dataset: GrammarRuleDataset,
    count: int,
    *,
    min_fitness: float | None,
    max_fitness: float | None,
) -> list[int]:
    indexed: list[tuple[float, int]] = []
    for idx, fitness in enumerate(dataset.valid_fitnesses):
        try:
            value = float(fitness)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value) and not math.isinf(value) and _fitness_in_range(value, min_fitness, max_fitness):
            indexed.append((value, idx))
    indexed.sort(reverse=True)
    return [idx for _, idx in indexed[:count]]


def _tensor_for_genotype(genotype: str, max_length: int) -> torch.Tensor | None:
    try:
        return encode_grammar_rule_string("".join(genotype.split()), max_length)
    except Exception:
        return None


def _make_seed_entries(
    *,
    seed_source: str,
    dataset: GrammarRuleDataset,
    max_length: int,
    count: int,
    evaluator: FramsticksFitness,
    rng: random.Random,
    mutation_pool_size: int,
    mutation_attempts: int,
    min_fitness: float | None,
    max_fitness: float | None,
) -> list[dict[str, Any]]:
    if seed_source == "dataset_top":
        entries: list[dict[str, Any]] = []
        for rank, dataset_idx in enumerate(
            _top_dataset_indices(dataset, count, min_fitness=min_fitness, max_fitness=max_fitness),
            start=1,
        ):
            genotype = dataset.valid_lines[dataset_idx]
            entries.append(
                {
                    "rank": rank,
                    "dataset_idx": dataset_idx,
                    "genotype": genotype,
                    "fitness": float(dataset.valid_fitnesses[dataset_idx]),
                    "source": "dataset_top",
                    "tensor": dataset[dataset_idx].unsqueeze(0),
                }
            )
        if not entries:
            raise RuntimeError("No dataset seeds matched the requested fitness range")
        return entries

    if seed_source == "simplest":
        genotype = evaluator._ensure_loaded().getSimplest("1")
        tensor = _tensor_for_genotype(genotype, max_length)
        if tensor is None:
            raise ValueError(f"Simplest genotype cannot be encoded by current grammar: {genotype}")
        fitness = evaluator.evaluate_one(genotype)
        seed_fitness = INVALID_FITNESS if fitness is None else float(fitness)
        if not _fitness_in_range(seed_fitness, min_fitness, max_fitness):
            raise RuntimeError(f"Simplest seed fitness {seed_fitness} is outside requested range")
        return [
            {
                "rank": 1,
                "dataset_idx": "",
                "genotype": genotype,
                "fitness": seed_fitness,
                "source": "simplest",
                "tensor": tensor,
            }
        ]

    frams_lib = evaluator._ensure_loaded()
    base = frams_lib.getSimplest("1")
    candidates = [base]
    valid: set[str] = set()
    attempts = 0
    while attempts < mutation_attempts and len(valid) < mutation_pool_size:
        parent = rng.choice(candidates)
        attempts += 1
        try:
            child = frams_lib.mutate([parent])[0]
        except Exception:
            continue
        child = "".join(child.split())
        candidates.append(child)
        if _is_valid_f1(child) and _tensor_for_genotype(child, max_length) is not None:
            valid.add(child)

    valid_list = sorted(valid)
    scores = evaluator.evaluate_many(valid_list) if valid_list else []
    scored: list[tuple[float, str]] = []
    for genotype, score in zip(valid_list, scores):
        if score is not None and _fitness_in_range(float(score), min_fitness, max_fitness):
            scored.append((float(score), genotype))
    scored.sort(reverse=True)

    entries = []
    for rank, (fitness, genotype) in enumerate(scored[:count], start=1):
        tensor = _tensor_for_genotype(genotype, max_length)
        if tensor is None:
            continue
        entries.append(
            {
                "rank": rank,
                "dataset_idx": "",
                "genotype": genotype,
                "fitness": fitness,
                "source": "framsticks_mutations",
                "tensor": tensor,
            }
        )
    if not entries:
        raise RuntimeError("No encodable f1 seeds generated by Framsticks mutations")
    return entries


def _score_genotypes(
    genotypes: list[str],
    evaluator: FramsticksFitness,
    cache: dict[str, float],
) -> list[float]:
    scores: list[float | None] = []
    to_eval: list[str] = []
    for genotype in genotypes:
        normalized = "".join(genotype.split())
        if not _is_valid_f1(normalized):
            scores.append(INVALID_FITNESS)
        elif normalized in cache:
            scores.append(cache[normalized])
        else:
            scores.append(None)
            if normalized not in to_eval:
                to_eval.append(normalized)

    if to_eval:
        values = evaluator.evaluate_many(to_eval)
        if len(values) != len(to_eval):
            values = [evaluator.evaluate_one(genotype) for genotype in to_eval]
        for genotype, value in zip(to_eval, values):
            cache[genotype] = INVALID_FITNESS if value is None else float(value)

    out: list[float] = []
    for genotype, score in zip(genotypes, scores):
        if score is not None:
            out.append(float(score))
        else:
            out.append(cache["".join(genotype.split())])
    return out


def _normalize_genotype(genotype: str) -> str:
    return "".join(genotype.split())


def _seeded_cem(
    *,
    model_name: str,
    model,
    seed_z: np.ndarray,
    evaluator: FramsticksFitness,
    device: torch.device,
    rng: np.random.Generator,
    iterations: int,
    population_size: int,
    elite_fraction: float,
    initial_std: float,
    smoothing: float,
    min_std: float,
    score_cache: dict[str, float],
    deadline: float | None,
    condition_value: float | None,
) -> dict[str, Any]:
    mean = seed_z.astype(np.float64).copy()
    std = np.full_like(mean, initial_std, dtype=np.float64)
    elite_count = max(2, int(population_size * elite_fraction))
    best_score = -np.inf
    best_genotype = ""
    best_z = mean.copy()
    history: list[float] = []

    for _ in range(iterations):
        if deadline is not None and history and time.monotonic() >= deadline:
            break
        population = rng.normal(mean, std, size=(population_size, mean.size))
        population[0] = mean
        genotypes = _decode_many(model_name, model, population, device, condition_value)
        scores = np.asarray(_score_genotypes(genotypes, evaluator, score_cache), dtype=np.float64)
        order = np.argsort(scores)[::-1]
        if float(scores[order[0]]) > best_score:
            best_score = float(scores[order[0]])
            best_genotype = genotypes[int(order[0])]
            best_z = population[int(order[0])].copy()

        elite = population[order[:elite_count]]
        elite_mean = elite.mean(axis=0)
        elite_std = elite.std(axis=0)
        mean = smoothing * mean + (1.0 - smoothing) * elite_mean
        std = smoothing * std + (1.0 - smoothing) * elite_std
        std = np.maximum(std, min_std)
        history.append(best_score)

    return {
        "best_score": best_score,
        "best_genotype": best_genotype,
        "best_z": best_z.tolist(),
        "history": history,
    }


def _seeded_cmaes(
    *,
    model_name: str,
    model,
    seed_z: np.ndarray,
    evaluator: FramsticksFitness,
    device: torch.device,
    rng: np.random.Generator,
    iterations: int,
    population_size: int,
    sigma: float,
    score_cache: dict[str, float],
    deadline: float | None,
    condition_value: float | None,
) -> dict[str, Any]:
    n = seed_z.size
    lamb = population_size
    mu = max(2, lamb // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights = weights / np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)

    cc = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    cs = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    c1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    cmu = min(1.0 - c1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
    damps = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + cs
    chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

    mean = seed_z.astype(np.float64).copy()
    cov = np.eye(n, dtype=np.float64)
    p_c = np.zeros(n, dtype=np.float64)
    p_s = np.zeros(n, dtype=np.float64)
    best_score = -np.inf
    best_genotype = ""
    best_z = mean.copy()
    history: list[float] = []

    for gen in range(iterations):
        if deadline is not None and history and time.monotonic() >= deadline:
            break
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        inv_sqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        y_samples = rng.standard_normal((lamb, n)) @ sqrt_cov.T
        population = mean + sigma * y_samples
        population[0] = mean

        genotypes = _decode_many(model_name, model, population, device, condition_value)
        scores = np.asarray(_score_genotypes(genotypes, evaluator, score_cache), dtype=np.float64)
        order = np.argsort(scores)[::-1]
        if float(scores[order[0]]) > best_score:
            best_score = float(scores[order[0]])
            best_genotype = genotypes[int(order[0])]
            best_z = population[int(order[0])].copy()

        x_selected = population[order[:mu]]
        mean_old = mean.copy()
        mean = np.sum(x_selected * weights[:, None], axis=0)
        y_w = (mean - mean_old) / max(sigma, 1e-12)
        p_s = (1.0 - cs) * p_s + np.sqrt(cs * (2.0 - cs) * mu_eff) * (inv_sqrt_cov @ y_w)
        norm_ps = np.linalg.norm(p_s)
        hsig = float(norm_ps / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (gen + 1.0))) / chi_n < (1.4 + 2.0 / (n + 1.0)))
        p_c = (1.0 - cc) * p_c + hsig * np.sqrt(cc * (2.0 - cc) * mu_eff) * y_w
        artmp = (x_selected - mean_old) / max(sigma, 1e-12)
        rank_mu = np.zeros_like(cov)
        for i in range(mu):
            rank_mu += weights[i] * np.outer(artmp[i], artmp[i])
        cov = (1.0 - c1 - cmu) * cov + c1 * (np.outer(p_c, p_c) + (1.0 - hsig) * cc * (2.0 - cc) * cov) + cmu * rank_mu
        cov = 0.5 * (cov + cov.T)
        sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1.0)))
        history.append(best_score)

    return {
        "best_score": best_score,
        "best_genotype": best_genotype,
        "best_z": best_z.tolist(),
        "history": history,
    }


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    if value is None:
        return ""
    return value


def main() -> None:
    args = _build_parser().parse_args()
    checkpoints_root = Path(maybe_resolve_path(args.checkpoints_root, root_dir=str(ROOT)))
    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    rng = np.random.default_rng(args.seed)

    evaluator = FramsticksFitness(
        maybe_resolve_path(args.framsticks_path, root_dir=str(ROOT)),
        lib=args.framsticks_lib,
        sim=maybe_resolve_path(args.framsticks_sim, root_dir=str(ROOT)),
    )
    evaluator.evaluate_one("X")

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"runs": {}}

    for label in labels:
        run_dir = checkpoints_root / label
        print(f"Optimizing {label}", flush=True)
        checkpoint_path = _find_checkpoint(run_dir)
        run_config = _load_json(run_dir / "run_config.json")
        state_dict, checkpoint_meta = load_checkpoint(str(checkpoint_path), device)
        cfg = _model_config(run_config, checkpoint_meta)
        model_name = cfg["model_name"]
        condition_value = None
        if args.condition_fitness is not None:
            fitness_mean = checkpoint_meta.get("fitness_mean", run_config.get("fitness_mean"))
            fitness_std = checkpoint_meta.get("fitness_std", run_config.get("fitness_std"))
            if fitness_mean is not None and fitness_std is not None:
                condition_value = (args.condition_fitness - float(fitness_mean)) / max(float(fitness_std), 1e-8)
            else:
                condition_value = args.condition_fitness
        dataset_cls = dataset_class_for_model(model_name)
        dataset = dataset_cls(data_path, cfg["max_length"])
        if not isinstance(dataset, GrammarRuleDataset):
            raise TypeError(f"Only grammar-rule datasets are supported, got {type(dataset).__name__}")
        known_genotypes = {_normalize_genotype(genotype) for genotype in dataset.valid_lines}

        model = build_model(
            model_name=model_name,
            dataset=dataset,
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            embedding_dim=cfg["embedding_dim"],
            max_length=cfg["max_length"],
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        py_rng = random.Random(args.seed)
        seed_entries = _make_seed_entries(
            seed_source=args.seed_source,
            dataset=dataset,
            max_length=cfg["max_length"],
            count=args.seed_candidates,
            evaluator=evaluator,
            rng=py_rng,
            mutation_pool_size=args.mutation_pool_size,
            mutation_attempts=args.mutation_attempts,
            min_fitness=args.seed_min_fitness,
            max_fitness=args.seed_max_fitness,
        )
        score_cache: dict[str, float] = {}
        run_details = []
        best_row: dict[str, Any] | None = None

        deadline = time.monotonic() + args.time_budget_seconds if args.time_budget_seconds is not None else None
        per_seed_budget = None
        if args.split_time_budget_across_seeds and args.time_budget_seconds is not None:
            per_seed_budget = args.time_budget_seconds / max(1, len(seed_entries))
        with torch.no_grad():
            for seed_entry in seed_entries:
                if deadline is not None and best_row is not None and time.monotonic() >= deadline:
                    break
                seed_deadline = deadline
                if deadline is not None and per_seed_budget is not None:
                    seed_deadline = min(deadline, time.monotonic() + per_seed_budget)
                rank = seed_entry["rank"]
                dataset_idx = seed_entry["dataset_idx"]
                genotype = seed_entry["genotype"]
                dataset_fitness = float(seed_entry["fitness"])
                batch = seed_entry["tensor"].to(device)
                seed_z = _encode_mu(model_name, model, batch).squeeze(0).detach().cpu().numpy()
                seed_decoded = _decode_from_z(model_name, model, seed_z, device, condition_value)
                seed_decoded_score = _score_genotypes([seed_decoded], evaluator, score_cache)[0]

                if args.algorithm == "cmaes":
                    result = _seeded_cmaes(
                        model_name=model_name,
                        model=model,
                        seed_z=seed_z,
                        evaluator=evaluator,
                        device=device,
                        rng=rng,
                        iterations=args.iterations,
                        population_size=args.population_size,
                        sigma=args.cma_sigma if args.cma_sigma is not None else args.initial_std,
                        score_cache=score_cache,
                        deadline=seed_deadline,
                        condition_value=condition_value,
                    )
                else:
                    result = _seeded_cem(
                        model_name=model_name,
                        model=model,
                        seed_z=seed_z,
                        evaluator=evaluator,
                        device=device,
                        rng=rng,
                        iterations=args.iterations,
                        population_size=args.population_size,
                        elite_fraction=args.elite_fraction,
                        initial_std=args.initial_std,
                        smoothing=args.smoothing,
                        min_std=args.min_std,
                        score_cache=score_cache,
                        deadline=seed_deadline,
                        condition_value=condition_value,
                    )
                best_score = float(result["best_score"])
                best_genotype = str(result["best_genotype"])
                best_normalized = _normalize_genotype(best_genotype)
                seed_normalized = _normalize_genotype(genotype)
                seed_decoded_normalized = _normalize_genotype(seed_decoded)
                row = {
                    "label": label,
                    "model": model_name,
                    "algorithm": args.algorithm,
                    "seed_source": args.seed_source,
                    "checkpoint": checkpoint_path.relative_to(run_dir).as_posix(),
                    "seed_rank": rank,
                    "seed_dataset_idx": dataset_idx,
                    "seed_source_fitness": dataset_fitness,
                    "seed_genotype": genotype,
                    "seed_decoded_fitness": seed_decoded_score,
                    "seed_decoded_genotype": seed_decoded,
                    "best_score": best_score,
                    "best_genotype": best_genotype,
                    "best_in_dataset": best_normalized in known_genotypes,
                    "best_is_source_seed": best_normalized == seed_normalized,
                    "best_is_decoded_seed": best_normalized == seed_decoded_normalized,
                    "best_minus_seed_source": best_score - dataset_fitness,
                    "best_minus_seed_decoded": best_score - seed_decoded_score,
                    "evaluated_unique": len(score_cache),
                    "iterations": args.iterations,
                    "actual_iterations": len(result["history"]),
                    "population_size": args.population_size,
                    "initial_std": args.initial_std,
                    "seed_min_fitness": args.seed_min_fitness,
                    "seed_max_fitness": args.seed_max_fitness,
                    "time_budget_seconds": args.time_budget_seconds,
                    "split_time_budget_across_seeds": args.split_time_budget_across_seeds,
                    "per_seed_budget_seconds": per_seed_budget,
                    "condition_fitness": args.condition_fitness,
                    "condition_value": condition_value,
                }
                rows.append(row)
                run_details.append({**row, "history": result["history"], "best_z": result["best_z"]})
                if best_row is None or float(row["best_score"]) > float(best_row["best_score"]):
                    best_row = row
                print(
                    f"  seed {rank}: source={dataset_fitness:.6g} "
                    f"decoded={seed_decoded_score:.6g} best={result['best_score']:.6g}",
                    flush=True,
                )

        details["runs"][label] = run_details
        if best_row is not None:
            print(f"Best {label}: {best_row['best_score']:.6g} {best_row['best_genotype']}", flush=True)

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = checkpoints_root / f"latent_optimization_{stamp}.csv"
    else:
        output_path = Path(maybe_resolve_path(args.output, root_dir=str(ROOT)))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in fieldnames})

    details_path = output_path.with_suffix(".details.json")
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}", flush=True)
    print(f"Wrote: {details_path}", flush=True)


if __name__ == "__main__":
    main()
