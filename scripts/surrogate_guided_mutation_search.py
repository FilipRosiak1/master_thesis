from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1vae.config.io import maybe_resolve_path
from f1vae.features import FEATURE_NAMES, featurize_genotypes
from f1vae.fitness.framsticks import FramsticksFitness, INVALID_FITNESS

try:
    from optimize_selected_latents import _is_valid_f1, _tensor_for_genotype
except Exception:
    _is_valid_f1 = None
    _tensor_for_genotype = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use a structural surrogate to filter Framsticks mutations")
    parser.add_argument("--surrogate", default="models/f1_surrogate/structural_ensemble/surrogate.pkl")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--framsticks-path", default="src/framsticks/Framsticks54")
    parser.add_argument("--framsticks-lib", default=None)
    parser.add_argument("--framsticks-sim", default="src/framsticks/framspy/eval-allcriteria.sim")
    parser.add_argument("--seed-source", choices=("dataset_top", "simplest"), default="dataset_top")
    parser.add_argument("--seed-min-fitness", type=float, default=None)
    parser.add_argument("--seed-max-fitness", type=float, default=1.2)
    parser.add_argument("--seed-candidates", type=int, default=5)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=1024)
    parser.add_argument("--true-evals-per-generation", type=int, default=32)
    parser.add_argument("--parents", type=int, default=8)
    parser.add_argument("--selection-modes", default="surrogate,random")
    parser.add_argument("--time-budget-seconds", type=float, default=None, help="Optional wall-clock budget for the whole search")
    parser.add_argument("--split-time-budget-across-runs", action="store_true", help="Split the budget across seed/mode runs")
    parser.add_argument("--uncertainty-weight", type=float, default=0.25)
    parser.add_argument("--length-penalty-weight", type=float, default=0.05)
    parser.add_argument("--length-penalty-radius", type=float, default=2.0)
    parser.add_argument("--ood-penalty-weight", type=float, default=0.02)
    parser.add_argument("--ood-penalty-radius", type=float, default=3.0)
    parser.add_argument("--mutation-attempt-factor", type=int, default=20)
    parser.add_argument("--require-grammar", action="store_true")
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output", default=None)
    parser.add_argument("--history-output", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize(genotype: str) -> str:
    return "".join(str(genotype).split())


def _read_dataset(path: str) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row or "\t" not in row:
                continue
            genotype, raw_fitness = row.split("\t", 1)
            try:
                fitness = float(raw_fitness.strip())
            except ValueError:
                continue
            if math.isnan(fitness) or math.isinf(fitness):
                continue
            rows.append((_normalize(genotype), fitness))
    return rows


def _fitness_in_range(value: float, min_fitness: float | None, max_fitness: float | None) -> bool:
    if min_fitness is not None and value < min_fitness:
        return False
    if max_fitness is not None and value > max_fitness:
        return False
    return True


def _seed_entries(rows: list[tuple[str, float]], count: int, min_fitness: float | None, max_fitness: float | None) -> list[dict[str, Any]]:
    filtered = [(fitness, genotype, idx) for idx, (genotype, fitness) in enumerate(rows) if _fitness_in_range(fitness, min_fitness, max_fitness)]
    filtered.sort(reverse=True)
    return [
        {"seed_rank": rank, "seed_dataset_idx": idx, "seed_genotype": genotype, "seed_fitness": fitness}
        for rank, (fitness, genotype, idx) in enumerate(filtered[:count], start=1)
    ]


def _simplest_seed_entry(frams_lib, evaluator: FramsticksFitness, min_fitness: float | None, max_fitness: float | None) -> list[dict[str, Any]]:
    genotype = _normalize(frams_lib.getSimplest("1"))
    fitness = evaluator.evaluate_one(genotype)
    seed_fitness = INVALID_FITNESS if fitness is None else float(fitness)
    if not _fitness_in_range(seed_fitness, min_fitness, max_fitness):
        raise RuntimeError(f"Simplest seed fitness {seed_fitness} is outside requested range")
    return [
        {
            "seed_rank": 1,
            "seed_dataset_idx": "",
            "seed_genotype": genotype,
            "seed_fitness": seed_fitness,
        }
    ]


def _load_surrogate(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if "models" not in payload:
        raise ValueError(f"Invalid surrogate payload: {path}")
    return payload


def _predict(payload: dict[str, Any], genotypes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = featurize_genotypes(genotypes)
    predictions = np.vstack([model.predict(x) for model in payload["models"]])
    return predictions.mean(axis=0), predictions.std(axis=0), x


def _acquisition(
    payload: dict[str, Any],
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    features: np.ndarray,
    *,
    uncertainty_weight: float,
    length_penalty_weight: float,
    length_penalty_radius: float,
    ood_penalty_weight: float,
    ood_penalty_radius: float,
) -> np.ndarray:
    length_idx = FEATURE_NAMES.index("length")
    length_z = np.abs((features[:, length_idx] - float(payload["length_mean"])) / max(float(payload["length_std"]), 1e-8))
    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
    ood_z = np.mean(np.abs((features - feature_mean) / np.maximum(feature_std, 1e-8)), axis=1)
    return (
        pred_mean
        - uncertainty_weight * pred_std
        - length_penalty_weight * np.maximum(0.0, length_z - length_penalty_radius)
        - ood_penalty_weight * np.maximum(0.0, ood_z - ood_penalty_radius)
    )


def _grammar_ok(genotype: str, max_length: int) -> bool:
    if _is_valid_f1 is None or _tensor_for_genotype is None:
        return True
    return bool(_is_valid_f1(genotype) and _tensor_for_genotype(genotype, max_length) is not None)


def _mutate_pool(
    frams_lib,
    parents: list[str],
    rng: random.Random,
    pool_size: int,
    attempt_factor: int,
    require_grammar: bool,
    max_length: int,
) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set(parents)
    attempts = 0
    max_attempts = max(pool_size, pool_size * attempt_factor)
    while len(candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        parent = rng.choice(parents)
        try:
            child = _normalize(frams_lib.mutate([parent])[0])
        except Exception:
            continue
        if not child or child in seen:
            continue
        if require_grammar and not _grammar_ok(child, max_length):
            continue
        seen.add(child)
        candidates.append(child)
    return candidates


def _evaluate_true(genotypes: list[str], evaluator: FramsticksFitness, cache: dict[str, float]) -> list[float]:
    to_eval: list[str] = []
    for genotype in genotypes:
        genotype = _normalize(genotype)
        if genotype not in cache and genotype not in to_eval:
            to_eval.append(genotype)
    if to_eval:
        values = evaluator.evaluate_many(to_eval)
        if len(values) != len(to_eval):
            values = [evaluator.evaluate_one(genotype) for genotype in to_eval]
        for genotype, value in zip(to_eval, values):
            cache[genotype] = INVALID_FITNESS if value is None else float(value)
    return [cache[_normalize(genotype)] for genotype in genotypes]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in fieldnames})


def _format_cell(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    if value is None:
        return ""
    return value


def _plot_history(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is unavailable; skipped progress plot", flush=True)
        return False
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["seed_rank"]), str(row["selection_mode"])), []).append(row)
    fig, ax = plt.subplots(figsize=(12, 7))
    for (seed_rank, selection_mode), group in sorted(grouped.items()):
        group.sort(key=lambda item: int(item["generation"]))
        ax.plot(
            [int(item["generation"]) for item in group],
            [float(item["best_true_fitness"] ) for item in group],
            label=f"seed {seed_rank} {selection_mode}",
            linewidth=1.4,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best true vertpos so far")
    ax.set_title("Surrogate-guided mutation search")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = _build_parser().parse_args()
    rng = random.Random(args.seed)
    surrogate = _load_surrogate(_resolve(args.surrogate))
    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    dataset_rows = _read_dataset(data_path)
    known = {genotype for genotype, _ in dataset_rows}
    selection_modes = _split_csv(args.selection_modes)
    invalid_modes = sorted(set(selection_modes) - {"surrogate", "random"})
    if invalid_modes:
        raise ValueError(f"Unsupported selection modes: {', '.join(invalid_modes)}")

    evaluator = FramsticksFitness(
        maybe_resolve_path(args.framsticks_path, root_dir=str(ROOT)),
        lib=args.framsticks_lib,
        sim=maybe_resolve_path(args.framsticks_sim, root_dir=str(ROOT)),
    )
    evaluator.evaluate_one("X")
    frams_lib = evaluator._ensure_loaded()
    if args.seed_source == "simplest":
        seeds = _simplest_seed_entry(frams_lib, evaluator, args.seed_min_fitness, args.seed_max_fitness)
    else:
        seeds = _seed_entries(dataset_rows, args.seed_candidates, args.seed_min_fitness, args.seed_max_fitness)
    if not seeds:
        raise RuntimeError("No seeds matched the requested fitness range")

    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"args": vars(args), "seeds": seeds, "runs": []}
    deadline = time.monotonic() + args.time_budget_seconds if args.time_budget_seconds is not None else None
    per_run_budget = None
    if args.split_time_budget_across_runs and args.time_budget_seconds is not None:
        per_run_budget = args.time_budget_seconds / max(1, len(seeds) * len(selection_modes))

    for seed in seeds:
        if deadline is not None and time.monotonic() >= deadline:
            break
        seed_rank = int(seed["seed_rank"])
        seed_genotype = str(seed["seed_genotype"])
        seed_fitness = float(seed["seed_fitness"])
        print(f"Seed {seed_rank}: {seed_fitness:.6g} {seed_genotype}", flush=True)
        for selection_mode in selection_modes:
            if deadline is not None and time.monotonic() >= deadline:
                break
            run_deadline = deadline
            if per_run_budget is not None:
                run_deadline = min(deadline, time.monotonic() + per_run_budget) if deadline is not None else time.monotonic() + per_run_budget
            true_cache: dict[str, float] = {seed_genotype: seed_fitness}
            parents = [seed_genotype]
            best_genotype = seed_genotype
            best_true = seed_fitness
            mode_rows: list[dict[str, Any]] = []
            for generation in range(1, args.generations + 1):
                if run_deadline is not None and time.monotonic() >= run_deadline:
                    print(f"  {selection_mode}: time budget reached before generation {generation}", flush=True)
                    break
                candidates = _mutate_pool(
                    frams_lib,
                    parents,
                    rng,
                    args.pool_size,
                    args.mutation_attempt_factor,
                    args.require_grammar,
                    args.max_length,
                )
                if not candidates:
                    history_rows.append(
                        {
                            "seed_rank": seed_rank,
                            "selection_mode": selection_mode,
                            "generation": generation,
                            "best_true_fitness": best_true,
                            "evaluated_unique": len(true_cache),
                            "candidate_pool": 0,
                        }
                    )
                    continue
                pred_mean, pred_std, features = _predict(surrogate, candidates)
                acquisition = _acquisition(
                    surrogate,
                    pred_mean,
                    pred_std,
                    features,
                    uncertainty_weight=args.uncertainty_weight,
                    length_penalty_weight=args.length_penalty_weight,
                    length_penalty_radius=args.length_penalty_radius,
                    ood_penalty_weight=args.ood_penalty_weight,
                    ood_penalty_radius=args.ood_penalty_radius,
                )
                eval_count = min(args.true_evals_per_generation, len(candidates))
                if selection_mode == "surrogate":
                    selected_idx = np.argsort(acquisition)[-eval_count:][::-1]
                else:
                    selected_idx = np.asarray(rng.sample(range(len(candidates)), eval_count), dtype=np.int64)
                selected = [candidates[int(idx)] for idx in selected_idx]
                true_scores = _evaluate_true(selected, evaluator, true_cache)
                for local_idx, genotype, true_score in zip(selected_idx, selected, true_scores):
                    local_idx = int(local_idx)
                    if true_score > best_true:
                        best_true = true_score
                        best_genotype = genotype
                    row = {
                        "seed_rank": seed_rank,
                        "seed_dataset_idx": seed["seed_dataset_idx"],
                        "seed_fitness": seed_fitness,
                        "seed_genotype": seed_genotype,
                        "selection_mode": selection_mode,
                        "generation": generation,
                        "genotype": genotype,
                        "true_fitness": true_score,
                        "pred_mean": float(pred_mean[local_idx]),
                        "pred_std": float(pred_std[local_idx]),
                        "acquisition": float(acquisition[local_idx]),
                        "best_true_fitness": best_true,
                        "best_genotype": best_genotype,
                        "candidate_pool": len(candidates),
                        "evaluated_unique": len(true_cache),
                        "in_dataset": genotype in known,
                        "is_seed": genotype == seed_genotype,
                        "improvement_over_seed": true_score - seed_fitness,
                        "best_improvement_over_seed": best_true - seed_fitness,
                    }
                    rows.append(row)
                    mode_rows.append(row)
                scored_parents = sorted(true_cache.items(), key=lambda item: item[1], reverse=True)
                parents = [genotype for genotype, _ in scored_parents[: max(1, args.parents)]]
                history_rows.append(
                    {
                        "seed_rank": seed_rank,
                        "selection_mode": selection_mode,
                        "generation": generation,
                        "best_true_fitness": best_true,
                        "best_improvement_over_seed": best_true - seed_fitness,
                        "evaluated_unique": len(true_cache),
                        "candidate_pool": len(candidates),
                    }
                )
                print(
                    f"  {selection_mode} gen {generation}: best={best_true:.6g} evals={len(true_cache)} pool={len(candidates)}",
                    flush=True,
                )
            details["runs"].append(
                {
                    "seed_rank": seed_rank,
                    "selection_mode": selection_mode,
                    "best_true_fitness": best_true,
                    "best_genotype": best_genotype,
                    "rows": mode_rows,
                }
            )

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _resolve(f"exports/surrogate_guided_mutation_{stamp}.csv")
    else:
        output_path = _resolve(args.output)
    history_path = _resolve(args.history_output) if args.history_output else output_path.with_name(f"{output_path.stem}.history.csv")
    plot_path = _resolve(args.plot_output) if args.plot_output else output_path.with_name(f"{output_path.stem}.history.png")
    details_path = output_path.with_suffix(".details.json")

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
