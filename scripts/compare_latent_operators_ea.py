from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1vae.config.io import maybe_resolve_path
from f1vae.data.datasets import GrammarRuleDataset
from f1vae.fitness.framsticks import FramsticksFitness, INVALID_FITNESS
from f1vae.inference.checkpoints import load_checkpoint
from f1vae.models.registry import build_model, dataset_class_for_model

from optimize_selected_latents import (
    _condition_value_for_model,
    _decode_from_z,
    _decode_many,
    _encode_mu,
    _find_checkpoint,
    _is_valid_f1,
    _load_json,
    _make_seed_entries,
    _model_config,
    _normalize_genotype,
    _score_genotypes,
)


DEFAULT_LABELS = "09_tree_vae_epoch80,10_transformer_vae_epoch90"


@dataclass
class Individual:
    genotype: str
    fitness: float
    z: np.ndarray | None = None
    source: str = ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare standard Framsticks operators against genotype-level EA using latent-space mutation/crossover"
    )
    parser.add_argument("--checkpoints-root", default="exports/f1_selected_10_ckpts_20260520_184738")
    parser.add_argument("--labels", default=DEFAULT_LABELS, help="Comma-separated checkpoint labels for latent operators")
    parser.add_argument("--methods", default="frams,latent", help="Comma-separated subset of: frams,latent")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--framsticks-path", default="src/framsticks/Framsticks54")
    parser.add_argument("--framsticks-lib", default=None)
    parser.add_argument("--framsticks-sim", default="src/framsticks/framspy/eval-allcriteria.sim")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed-source", choices=("dataset_top", "simplest", "framsticks_mutations"), default="dataset_top")
    parser.add_argument("--seed-candidates", type=int, default=5)
    parser.add_argument("--seed-min-fitness", type=float, default=None)
    parser.add_argument("--seed-max-fitness", type=float, default=1.2)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--offspring-size", type=int, default=None, help="Defaults to population-size")
    parser.add_argument("--elite-size", type=int, default=4)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--max-evaluations", type=int, default=None, help="Optional unique true-evaluation cap per run")
    parser.add_argument("--time-budget-seconds", type=float, default=None, help="Optional wall-clock cap per label")
    parser.add_argument("--split-time-budget-across-runs", action="store_true")
    parser.add_argument("--frams-crossover-prob", type=float, default=0.25)
    parser.add_argument("--frams-mutate-after-crossover-prob", type=float, default=0.25)
    parser.add_argument("--latent-mutation-stds", default="0.12,0.25,0.5,0.9")
    parser.add_argument("--latent-crossover-prob", type=float, default=0.45)
    parser.add_argument("--latent-extrapolate-prob", type=float, default=0.2)
    parser.add_argument("--latent-directional-prob", type=float, default=0.25)
    parser.add_argument("--latent-line-scale", type=float, default=0.6)
    parser.add_argument("--latent-random-immigrant-prob", type=float, default=0.05)
    parser.add_argument("--latent-reencode-offspring", action="store_true")
    parser.add_argument("--attempt-factor", type=int, default=30)
    parser.add_argument("--condition-fitness", type=float, default=None)
    parser.add_argument("--condition-length", type=float, default=None)
    parser.add_argument("--condition-segments", type=float, default=None)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--output", default=None)
    parser.add_argument("--history-output", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_list(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("At least one latent mutation std is required")
    if any(std < 0.0 for std in out):
        raise ValueError("Latent mutation std values must be non-negative")
    return out


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _format_cell(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in fieldnames})


def _known_dataset_genotypes(dataset: GrammarRuleDataset) -> set[str]:
    return {_normalize_genotype(genotype) for genotype in dataset.valid_lines}


def _evaluate_raw(genotypes: list[str], evaluator: FramsticksFitness, cache: dict[str, float]) -> list[float]:
    to_eval: list[str] = []
    for genotype in genotypes:
        normalized = _normalize_genotype(genotype)
        if normalized and normalized not in cache and normalized not in to_eval:
            to_eval.append(normalized)
    if to_eval:
        values = evaluator.evaluate_many(to_eval)
        if len(values) != len(to_eval):
            values = [evaluator.evaluate_one(genotype) for genotype in to_eval]
        for genotype, value in zip(to_eval, values):
            cache[genotype] = INVALID_FITNESS if value is None else float(value)
    return [cache.get(_normalize_genotype(genotype), INVALID_FITNESS) for genotype in genotypes]


def _tournament(population: list[Individual], rng: random.Random, size: int) -> Individual:
    size = max(1, min(size, len(population)))
    contestants = rng.sample(population, size)
    return max(contestants, key=lambda item: item.fitness)


def _best(population: list[Individual]) -> Individual:
    return max(population, key=lambda item: item.fitness)


def _select_survivors(population: list[Individual], population_size: int) -> list[Individual]:
    best_by_genotype: dict[str, Individual] = {}
    for individual in population:
        genotype = _normalize_genotype(individual.genotype)
        previous = best_by_genotype.get(genotype)
        if previous is None or individual.fitness > previous.fitness:
            best_by_genotype[genotype] = individual
    return sorted(best_by_genotype.values(), key=lambda item: item.fitness, reverse=True)[:population_size]


def _evaluation_limit_reached(cache: dict[str, float], max_evaluations: int | None) -> bool:
    return max_evaluations is not None and len(cache) >= max_evaluations


def _remaining_evaluation_slots(cache: dict[str, float], max_evaluations: int | None, requested: int) -> int:
    if max_evaluations is None:
        return requested
    return max(0, min(requested, max_evaluations - len(cache)))


def _make_frams_child(frams_lib, parents: list[Individual], rng: random.Random, crossover_prob: float, mutate_after_crossover_prob: float) -> str:
    if len(parents) > 1 and rng.random() < crossover_prob:
        left, right = rng.sample(parents, 2)
        child = frams_lib.crossOver(left.genotype, right.genotype)
        if rng.random() < mutate_after_crossover_prob:
            child = frams_lib.mutate([child])[0]
        return _normalize_genotype(child)
    parent = rng.choice(parents)
    return _normalize_genotype(frams_lib.mutate([parent.genotype])[0])


def _initial_frams_population(
    *,
    seed_genotype: str,
    seed_fitness: float,
    evaluator: FramsticksFitness,
    rng: random.Random,
    population_size: int,
    cache: dict[str, float],
    max_evaluations: int | None,
    attempt_factor: int,
) -> list[Individual]:
    frams_lib = evaluator._ensure_loaded()
    seed_genotype = _normalize_genotype(seed_genotype)
    cache[seed_genotype] = seed_fitness
    population = [Individual(seed_genotype, seed_fitness, source="seed")]
    attempts = 0
    max_attempts = max(population_size, population_size * attempt_factor)
    while len(population) < population_size and attempts < max_attempts and not _evaluation_limit_reached(cache, max_evaluations):
        attempts += 1
        try:
            child = _normalize_genotype(frams_lib.mutate([seed_genotype])[0])
        except Exception:
            continue
        if not child or child in cache:
            continue
        score = _evaluate_raw([child], evaluator, cache)[0]
        if score <= INVALID_FITNESS:
            continue
        population.append(Individual(child, score, source="frams_init"))
    return population


def _frams_ea(
    *,
    seed_genotype: str,
    seed_fitness: float,
    evaluator: FramsticksFitness,
    rng: random.Random,
    generations: int,
    population_size: int,
    offspring_size: int,
    elite_size: int,
    tournament_size: int,
    max_evaluations: int | None,
    deadline: float | None,
    crossover_prob: float,
    mutate_after_crossover_prob: float,
    attempt_factor: int,
) -> dict[str, Any]:
    cache: dict[str, float] = {}
    population = _initial_frams_population(
        seed_genotype=seed_genotype,
        seed_fitness=seed_fitness,
        evaluator=evaluator,
        rng=rng,
        population_size=population_size,
        cache=cache,
        max_evaluations=max_evaluations,
        attempt_factor=attempt_factor,
    )
    frams_lib = evaluator._ensure_loaded()
    history: list[dict[str, Any]] = []
    attempts_total = 0

    for generation in range(1, generations + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        if _evaluation_limit_reached(cache, max_evaluations):
            break
        parents = sorted(population, key=lambda item: item.fitness, reverse=True)[: max(1, min(elite_size, len(population)))]
        children: list[Individual] = []
        attempts = 0
        max_attempts = max(offspring_size, offspring_size * attempt_factor)
        requested = _remaining_evaluation_slots(cache, max_evaluations, offspring_size)
        while len(children) < requested and attempts < max_attempts:
            attempts += 1
            attempts_total += 1
            selected = [_tournament(population, rng, tournament_size) for _ in range(max(2, len(parents)))]
            try:
                child = _make_frams_child(frams_lib, selected, rng, crossover_prob, mutate_after_crossover_prob)
            except Exception:
                continue
            if not child or child in cache:
                continue
            score = _evaluate_raw([child], evaluator, cache)[0]
            if score <= INVALID_FITNESS:
                continue
            children.append(Individual(child, score, source="frams_child"))
        population = _select_survivors(population + children, population_size)
        best = _best(population)
        history.append(
            {
                "generation": generation,
                "best_score": best.fitness,
                "evaluated_unique": len(cache),
                "population_size": len(population),
                "children": len(children),
                "attempts": attempts,
            }
        )
        print(f"    frams gen {generation}: best={best.fitness:.6g} evals={len(cache)} children={len(children)}", flush=True)
    best = _best(population)
    return {
        "best_score": best.fitness,
        "best_genotype": best.genotype,
        "history": history,
        "evaluated_unique": len(cache),
        "attempts_total": attempts_total,
        "final_population": [item.genotype for item in population],
    }


def _condition_for_seed(condition_value: Any, seed_fitness: float) -> Any:
    return condition_value


def _encode_genotype(model_name: str, model, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        return _encode_mu(model_name, model, tensor.to(device)).squeeze(0).detach().cpu().numpy()


def _initial_latent_population(
    *,
    model_name: str,
    model,
    seed_z: np.ndarray,
    seed_genotype: str,
    seed_fitness: float,
    evaluator: FramsticksFitness,
    device: torch.device,
    rng_np: np.random.Generator,
    population_size: int,
    mutation_stds: list[float],
    cache: dict[str, float],
    max_evaluations: int | None,
    condition_value: Any,
    attempt_factor: int,
) -> list[Individual]:
    seed_genotype = _normalize_genotype(seed_genotype)
    cache[seed_genotype] = seed_fitness
    population = [Individual(seed_genotype, seed_fitness, seed_z.astype(np.float64).copy(), source="seed")]
    attempts = 0
    max_attempts = max(population_size, population_size * attempt_factor)
    while len(population) < population_size and attempts < max_attempts and not _evaluation_limit_reached(cache, max_evaluations):
        batch_count = min(population_size - len(population), 16)
        z_values = []
        for _ in range(batch_count):
            std = float(rng_np.choice(mutation_stds))
            z_values.append(seed_z + rng_np.normal(0.0, std, size=seed_z.size))
        attempts += len(z_values)
        genotypes = _decode_many(model_name, model, np.asarray(z_values), device, condition_value)
        slots = _remaining_evaluation_slots(cache, max_evaluations, len(genotypes))
        for genotype, z_value in zip(genotypes[:slots], z_values[:slots]):
            normalized = _normalize_genotype(genotype)
            if not normalized or normalized in cache or not _is_valid_f1(normalized):
                continue
            score = _score_genotypes([normalized], evaluator, cache)[0]
            if score <= INVALID_FITNESS:
                continue
            population.append(Individual(normalized, score, np.asarray(z_value, dtype=np.float64), source="latent_init"))
            if len(population) >= population_size:
                break
    return population


def _latent_child_z(
    population: list[Individual],
    rng: random.Random,
    rng_np: np.random.Generator,
    mutation_stds: list[float],
    tournament_size: int,
    crossover_prob: float,
    extrapolate_prob: float,
    directional_prob: float,
    line_scale: float,
    random_immigrant_prob: float,
) -> np.ndarray:
    z_values = [item.z for item in population if item.z is not None]
    if not z_values:
        raise RuntimeError("Latent population has no z values")
    dim = int(z_values[0].size)
    std = float(rng_np.choice(mutation_stds))
    if rng.random() < random_immigrant_prob:
        return rng_np.normal(0.0, 1.0, size=dim)

    parent = _tournament(population, rng, tournament_size)
    if parent.z is None:
        parent = max((item for item in population if item.z is not None), key=lambda item: item.fitness)
    z_parent = parent.z
    assert z_parent is not None

    if len(population) > 1 and rng.random() < directional_prob:
        best = max((item for item in population if item.z is not None), key=lambda item: item.fitness)
        other = _tournament(population, rng, tournament_size)
        if best.z is not None and other.z is not None:
            direction = best.z - other.z
            return best.z + line_scale * direction + rng_np.normal(0.0, std, size=dim)

    if len(population) > 1 and rng.random() < crossover_prob:
        mate = _tournament(population, rng, tournament_size)
        if mate.z is not None:
            if rng.random() < extrapolate_prob:
                alpha = rng_np.uniform(-0.5, 1.5, size=dim)
            else:
                alpha = rng_np.uniform(0.0, 1.0, size=dim)
            child = alpha * z_parent + (1.0 - alpha) * mate.z
            return child + rng_np.normal(0.0, std, size=dim)

    return z_parent + rng_np.normal(0.0, std, size=dim)


def _latent_ea(
    *,
    model_name: str,
    model,
    seed_z: np.ndarray,
    seed_genotype: str,
    seed_fitness: float,
    evaluator: FramsticksFitness,
    device: torch.device,
    rng: random.Random,
    rng_np: np.random.Generator,
    generations: int,
    population_size: int,
    offspring_size: int,
    elite_size: int,
    tournament_size: int,
    max_evaluations: int | None,
    deadline: float | None,
    mutation_stds: list[float],
    crossover_prob: float,
    extrapolate_prob: float,
    directional_prob: float,
    line_scale: float,
    random_immigrant_prob: float,
    reencode_offspring: bool,
    max_length: int,
    condition_value: Any,
    attempt_factor: int,
) -> dict[str, Any]:
    cache: dict[str, float] = {}
    population = _initial_latent_population(
        model_name=model_name,
        model=model,
        seed_z=seed_z,
        seed_genotype=seed_genotype,
        seed_fitness=seed_fitness,
        evaluator=evaluator,
        device=device,
        rng_np=rng_np,
        population_size=population_size,
        mutation_stds=mutation_stds,
        cache=cache,
        max_evaluations=max_evaluations,
        condition_value=condition_value,
        attempt_factor=attempt_factor,
    )
    history: list[dict[str, Any]] = []
    attempts_total = 0
    duplicate_count = 0
    invalid_count = 0

    for generation in range(1, generations + 1):
        if deadline is not None and time.monotonic() >= deadline:
            break
        if _evaluation_limit_reached(cache, max_evaluations):
            break
        children: list[Individual] = []
        attempts = 0
        max_attempts = max(offspring_size, offspring_size * attempt_factor)
        requested = _remaining_evaluation_slots(cache, max_evaluations, offspring_size)
        while len(children) < requested and attempts < max_attempts:
            batch_count = min(requested - len(children), 32, max_attempts - attempts)
            z_values = [
                _latent_child_z(
                    population,
                    rng,
                    rng_np,
                    mutation_stds,
                    tournament_size,
                    crossover_prob,
                    extrapolate_prob,
                    directional_prob,
                    line_scale,
                    random_immigrant_prob,
                )
                for _ in range(batch_count)
            ]
            attempts += batch_count
            attempts_total += batch_count
            genotypes = _decode_many(model_name, model, np.asarray(z_values), device, condition_value)
            for genotype, z_value in zip(genotypes, z_values):
                normalized = _normalize_genotype(genotype)
                if not normalized or not _is_valid_f1(normalized):
                    invalid_count += 1
                    continue
                if normalized in cache:
                    duplicate_count += 1
                    continue
                score = _score_genotypes([normalized], evaluator, cache)[0]
                if score <= INVALID_FITNESS:
                    invalid_count += 1
                    continue
                child_z = np.asarray(z_value, dtype=np.float64)
                if reencode_offspring:
                    tensor = None
                    try:
                        from f1vae.inference.encode import encode_grammar_rule_string

                        tensor = encode_grammar_rule_string(normalized, max_length)
                    except Exception:
                        tensor = None
                    if tensor is not None:
                        child_z = _encode_genotype(model_name, model, tensor, device)
                children.append(Individual(normalized, score, child_z, source="latent_child"))
                if len(children) >= requested:
                    break
        elites = sorted(population, key=lambda item: item.fitness, reverse=True)[: max(1, min(elite_size, len(population)))]
        population = _select_survivors(elites + population + children, population_size)
        best = _best(population)
        history.append(
            {
                "generation": generation,
                "best_score": best.fitness,
                "evaluated_unique": len(cache),
                "population_size": len(population),
                "children": len(children),
                "attempts": attempts,
                "duplicates": duplicate_count,
                "invalid": invalid_count,
            }
        )
        print(f"    latent gen {generation}: best={best.fitness:.6g} evals={len(cache)} children={len(children)}", flush=True)
    best = _best(population)
    return {
        "best_score": best.fitness,
        "best_genotype": best.genotype,
        "history": history,
        "evaluated_unique": len(cache),
        "attempts_total": attempts_total,
        "duplicate_count": duplicate_count,
        "invalid_count": invalid_count,
        "final_population": [item.genotype for item in population],
    }


def _plot_history(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; skipped plot", flush=True)
        return False
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["label"]), str(row["method"]), str(row["seed_rank"]))
        grouped.setdefault(key, []).append(row)
    fig, ax = plt.subplots(figsize=(14, 8))
    for (label, method, seed_rank), group in sorted(grouped.items()):
        group.sort(key=lambda item: int(item["generation"]))
        ax.plot(
            [int(item["generation"]) for item in group],
            [float(item["best_score"]) for item in group],
            linewidth=1.2,
            label=f"{label} {method} seed {seed_rank}",
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best true vertpos")
    ax.set_title("Standard Framsticks operators vs latent operators")
    ax.grid(True, alpha=0.25)
    if len(grouped) <= 24:
        ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(rect=(0, 0, 0.78, 1))
    else:
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = _build_parser().parse_args()
    labels = _split_csv(args.labels)
    methods = _split_csv(args.methods)
    invalid_methods = sorted(set(methods) - {"frams", "latent"})
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {', '.join(invalid_methods)}")
    if not labels:
        raise ValueError("At least one label is required")
    offspring_size = args.population_size if args.offspring_size is None else args.offspring_size
    mutation_stds = _parse_float_list(args.latent_mutation_stds)

    checkpoints_root = _resolve(args.checkpoints_root)
    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = FramsticksFitness(
        maybe_resolve_path(args.framsticks_path, root_dir=str(ROOT)),
        lib=args.framsticks_lib,
        sim=maybe_resolve_path(args.framsticks_sim, root_dir=str(ROOT)),
    )
    evaluator.evaluate_one("X")

    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"args": vars(args), "labels": labels, "methods": methods, "runs": {}}

    for label_idx, label in enumerate(labels):
        run_dir = checkpoints_root / label
        print(f"Loading {label}", flush=True)
        checkpoint_path = _find_checkpoint(run_dir)
        run_config = _load_json(run_dir / "run_config.json")
        state_dict, checkpoint_meta = load_checkpoint(str(checkpoint_path), device)
        cfg = _model_config(run_config, checkpoint_meta)
        model_name = cfg["model_name"]
        dataset_cls = dataset_class_for_model(model_name)
        dataset = dataset_cls(data_path, cfg["max_length"])
        if not isinstance(dataset, GrammarRuleDataset):
            raise TypeError(f"Only grammar-rule datasets are supported, got {type(dataset).__name__}")
        known = _known_dataset_genotypes(dataset)
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
        condition_value = _condition_value_for_model(model, args, run_config, checkpoint_meta)

        seed_rng = random.Random(args.seed + label_idx * 1009)
        seed_entries = _make_seed_entries(
            seed_source=args.seed_source,
            dataset=dataset,
            max_length=cfg["max_length"],
            count=args.seed_candidates,
            evaluator=evaluator,
            rng=seed_rng,
            mutation_pool_size=200,
            mutation_attempts=2000,
            min_fitness=args.seed_min_fitness,
            max_fitness=args.seed_max_fitness,
        )
        label_deadline = time.monotonic() + args.time_budget_seconds if args.time_budget_seconds is not None else None
        per_run_budget = None
        if args.split_time_budget_across_runs and args.time_budget_seconds is not None:
            per_run_budget = args.time_budget_seconds / max(1, len(seed_entries) * len(methods))
        label_details: list[dict[str, Any]] = []

        for seed_entry in seed_entries:
            seed_rank = int(seed_entry["rank"])
            seed_genotype = str(seed_entry["genotype"])
            seed_fitness = float(seed_entry["fitness"])
            seed_z = _encode_genotype(model_name, model, seed_entry["tensor"], device)
            seed_decoded = _decode_from_z(model_name, model, seed_z, device, condition_value)
            seed_decoded_score = _score_genotypes([seed_decoded], evaluator, {})[0]
            print(f"  seed {seed_rank}: source={seed_fitness:.6g} decoded={seed_decoded_score:.6g}", flush=True)

            for method_idx, method in enumerate(methods):
                if label_deadline is not None and time.monotonic() >= label_deadline:
                    break
                run_deadline = label_deadline
                if per_run_budget is not None:
                    run_deadline = min(label_deadline, time.monotonic() + per_run_budget) if label_deadline is not None else time.monotonic() + per_run_budget
                run_seed = args.seed + label_idx * 100000 + seed_rank * 1000 + method_idx * 137
                rng = random.Random(run_seed)
                rng_np = np.random.default_rng(run_seed)
                print(f"    method {method}", flush=True)
                if method == "frams":
                    result = _frams_ea(
                        seed_genotype=seed_genotype,
                        seed_fitness=seed_fitness,
                        evaluator=evaluator,
                        rng=rng,
                        generations=args.generations,
                        population_size=args.population_size,
                        offspring_size=offspring_size,
                        elite_size=args.elite_size,
                        tournament_size=args.tournament_size,
                        max_evaluations=args.max_evaluations,
                        deadline=run_deadline,
                        crossover_prob=args.frams_crossover_prob,
                        mutate_after_crossover_prob=args.frams_mutate_after_crossover_prob,
                        attempt_factor=args.attempt_factor,
                    )
                else:
                    result = _latent_ea(
                        model_name=model_name,
                        model=model,
                        seed_z=seed_z,
                        seed_genotype=seed_genotype,
                        seed_fitness=seed_fitness,
                        evaluator=evaluator,
                        device=device,
                        rng=rng,
                        rng_np=rng_np,
                        generations=args.generations,
                        population_size=args.population_size,
                        offspring_size=offspring_size,
                        elite_size=args.elite_size,
                        tournament_size=args.tournament_size,
                        max_evaluations=args.max_evaluations,
                        deadline=run_deadline,
                        mutation_stds=mutation_stds,
                        crossover_prob=args.latent_crossover_prob,
                        extrapolate_prob=args.latent_extrapolate_prob,
                        directional_prob=args.latent_directional_prob,
                        line_scale=args.latent_line_scale,
                        random_immigrant_prob=args.latent_random_immigrant_prob,
                        reencode_offspring=args.latent_reencode_offspring,
                        max_length=cfg["max_length"],
                        condition_value=condition_value,
                        attempt_factor=args.attempt_factor,
                    )
                best_score = float(result["best_score"])
                best_genotype = str(result["best_genotype"])
                row = {
                    "label": label,
                    "model": model_name,
                    "method": method,
                    "checkpoint": checkpoint_path.relative_to(run_dir).as_posix(),
                    "seed_rank": seed_rank,
                    "seed_source": args.seed_source,
                    "seed_dataset_idx": seed_entry.get("dataset_idx", ""),
                    "seed_source_fitness": seed_fitness,
                    "seed_decoded_fitness": seed_decoded_score,
                    "seed_genotype": seed_genotype,
                    "seed_decoded_genotype": seed_decoded,
                    "best_score": best_score,
                    "best_genotype": best_genotype,
                    "best_in_dataset": _normalize_genotype(best_genotype) in known,
                    "best_is_source_seed": _normalize_genotype(best_genotype) == _normalize_genotype(seed_genotype),
                    "best_is_decoded_seed": _normalize_genotype(best_genotype) == _normalize_genotype(seed_decoded),
                    "best_minus_seed_source": best_score - seed_fitness,
                    "best_minus_seed_decoded": best_score - seed_decoded_score,
                    "evaluated_unique": result.get("evaluated_unique", ""),
                    "generations_requested": args.generations,
                    "generations_actual": len(result.get("history", [])),
                    "population_size": args.population_size,
                    "offspring_size": offspring_size,
                    "max_evaluations": args.max_evaluations,
                    "attempts_total": result.get("attempts_total", ""),
                    "duplicate_count": result.get("duplicate_count", ""),
                    "invalid_count": result.get("invalid_count", ""),
                    "condition_fitness": args.condition_fitness,
                    "condition_length": args.condition_length,
                    "condition_segments": args.condition_segments,
                    "condition_value": condition_value,
                }
                rows.append(row)
                label_details.append({**row, "history": result.get("history", []), "final_population": result.get("final_population", [])})
                for hist in result.get("history", []):
                    history_rows.append(
                        {
                            "label": label,
                            "model": model_name,
                            "method": method,
                            "seed_rank": seed_rank,
                            "seed_source_fitness": seed_fitness,
                            "seed_decoded_fitness": seed_decoded_score,
                            "generation": hist["generation"],
                            "best_score": hist["best_score"],
                            "best_minus_seed_source": float(hist["best_score"]) - seed_fitness,
                            "best_minus_seed_decoded": float(hist["best_score"]) - seed_decoded_score,
                            "evaluated_unique": hist.get("evaluated_unique", ""),
                            "children": hist.get("children", ""),
                            "attempts": hist.get("attempts", ""),
                            "duplicates": hist.get("duplicates", ""),
                            "invalid": hist.get("invalid", ""),
                        }
                    )
                print(f"    best={best_score:.6g} delta_source={best_score - seed_fitness:.6g}", flush=True)
        details["runs"][label] = label_details

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _resolve(f"exports/latent_operator_ea_compare_{stamp}.csv")
    else:
        output_path = _resolve(args.output)
    history_path = _resolve(args.history_output) if args.history_output else output_path.with_name(f"{output_path.stem}.history.csv")
    details_path = output_path.with_suffix(".details.json")
    plot_path = _resolve(args.plot_output) if args.plot_output else output_path.with_name(f"{output_path.stem}.history.png")

    _write_csv(output_path, rows)
    _write_csv(history_path, history_rows)
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    if not args.no_plot and _plot_history(plot_path, history_rows):
        print(f"Wrote: {plot_path}", flush=True)
    print(f"Wrote: {output_path}", flush=True)
    print(f"Wrote: {history_path}", flush=True)
    print(f"Wrote: {details_path}", flush=True)


if __name__ == "__main__":
    main()
