from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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
from f1vae.grammars import f1 as G
from f1vae.inference.decode import decode_grammar_indices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CEM directly over f1 grammar productions")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--framsticks-path", default="src/framsticks/Framsticks54")
    parser.add_argument("--framsticks-lib", default=None)
    parser.add_argument("--framsticks-sim", default="src/framsticks/framspy/eval-allcriteria.sim")
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--init-top-count", type=int, default=200)
    parser.add_argument("--init-min-fitness", type=float, default=None)
    parser.add_argument("--init-max-fitness", type=float, default=1.2)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=512)
    parser.add_argument("--elite-fraction", type=float, default=0.1)
    parser.add_argument("--smoothing", type=float, default=0.5)
    parser.add_argument("--min-prob", type=float, default=1e-4)
    parser.add_argument("--length-penalty-weight", type=float, default=0.0)
    parser.add_argument("--target-length", type=float, default=None)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output", default=None)
    parser.add_argument("--history-output", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


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


def _fitness_in_range(value: float, min_fitness: float | None, max_fitness: float | None) -> bool:
    if min_fitness is not None and value < min_fitness:
        return False
    if max_fitness is not None and value > max_fitness:
        return False
    return True


def _initial_sequences(dataset: GrammarRuleDataset, count: int, min_fitness: float | None, max_fitness: float | None) -> list[list[int]]:
    indexed: list[tuple[float, int]] = []
    for idx, fitness in enumerate(dataset.valid_fitnesses):
        try:
            value = float(fitness)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value) and not math.isinf(value) and _fitness_in_range(value, min_fitness, max_fitness):
            indexed.append((value, idx))
    indexed.sort(reverse=True)
    sequences: list[list[int]] = []
    for _, idx in indexed[:count]:
        row = dataset[idx].tolist()
        sequences.append([int(rule) for rule in row if int(rule) != dataset.pad_rule_idx])
    return sequences


def _lhs_rule_pairs(sequence: list[int], max_lhs: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    stack = [int(np.where(np.asarray(G.lhs_list) == str(G.GCFG.start()))[0][0])]
    for rule_idx in sequence:
        if not stack:
            break
        lhs_idx = stack.pop()
        if 0 <= lhs_idx < max_lhs:
            pairs.append((lhs_idx, int(rule_idx)))
        if 0 <= rule_idx < len(G.rhs_map):
            stack.extend(reversed(G.rhs_map[rule_idx]))
    return pairs


def _normalize_probs(counts: np.ndarray, masks: np.ndarray, min_prob: float) -> np.ndarray:
    probs = counts.astype(np.float64) + min_prob
    probs *= masks
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, np.maximum(row_sums, 1e-12), out=np.zeros_like(probs), where=row_sums > 0)
    for row in range(probs.shape[0]):
        if probs[row].sum() <= 0:
            valid = masks[row] > 0
            probs[row, valid] = 1.0 / max(1, int(valid.sum()))
    return probs


def _init_probs(dataset: GrammarRuleDataset, args) -> np.ndarray:
    masks = np.asarray(G.masks, dtype=np.float64)
    counts = masks * float(args.min_prob)
    sequences = _initial_sequences(dataset, args.init_top_count, args.init_min_fitness, args.init_max_fitness)
    if not sequences:
        counts = masks.copy()
    else:
        for sequence in sequences:
            for lhs_idx, rule_idx in _lhs_rule_pairs(sequence, masks.shape[0]):
                counts[lhs_idx, rule_idx] += 1.0
    return _normalize_probs(counts, masks, args.min_prob)


def _sample_sequence(probs: np.ndarray, rng: np.random.Generator, max_length: int) -> tuple[list[int], bool]:
    start_lhs_idx = int(np.where(np.asarray(G.lhs_list) == str(G.GCFG.start()))[0][0])
    stack = [start_lhs_idx]
    sequence: list[int] = []
    while stack and len(sequence) < max_length:
        lhs_idx = stack.pop()
        row_probs = probs[lhs_idx]
        rule_idx = int(rng.choice(np.arange(row_probs.size), p=row_probs))
        sequence.append(rule_idx)
        stack.extend(reversed(G.rhs_map[rule_idx]))
    return sequence, not stack


def _decode_sequence(sequence: list[int]) -> str:
    return decode_grammar_indices(torch.tensor(sequence, dtype=torch.long))


def _evaluate(genotypes: list[str], evaluator: FramsticksFitness, cache: dict[str, float]) -> list[float]:
    to_eval: list[str] = []
    for genotype in genotypes:
        if genotype not in cache and genotype not in to_eval:
            to_eval.append(genotype)
    if to_eval:
        values = evaluator.evaluate_many(to_eval)
        if len(values) != len(to_eval):
            values = [evaluator.evaluate_one(genotype) for genotype in to_eval]
        for genotype, value in zip(to_eval, values):
            cache[genotype] = INVALID_FITNESS if value is None else float(value)
    return [cache[genotype] for genotype in genotypes]


def _objective(score: float, genotype: str, args) -> float:
    if args.target_length is None or args.length_penalty_weight <= 0.0:
        return score
    return score - args.length_penalty_weight * abs(len(genotype) - args.target_length)


def _update_probs(probs: np.ndarray, elite_sequences: list[list[int]], args) -> np.ndarray:
    masks = np.asarray(G.masks, dtype=np.float64)
    counts = masks * float(args.min_prob)
    for sequence in elite_sequences:
        for lhs_idx, rule_idx in _lhs_rule_pairs(sequence, masks.shape[0]):
            counts[lhs_idx, rule_idx] += 1.0
    elite_probs = _normalize_probs(counts, masks, args.min_prob)
    updated = args.smoothing * probs + (1.0 - args.smoothing) * elite_probs
    return _normalize_probs(updated, masks, args.min_prob)


def _plot_history(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is unavailable; skipped progress plot", flush=True)
        return False
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([int(row["generation"]) for row in rows], [float(row["best_true_fitness"]) for row in rows], linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best true vertpos so far")
    ax.set_title("Grammar CEM search")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = _build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    dataset = GrammarRuleDataset(data_path, args.max_length)
    probs = _init_probs(dataset, args)
    evaluator = FramsticksFitness(
        maybe_resolve_path(args.framsticks_path, root_dir=str(ROOT)),
        lib=args.framsticks_lib,
        sim=maybe_resolve_path(args.framsticks_sim, root_dir=str(ROOT)),
    )
    evaluator.evaluate_one("X")

    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    cache: dict[str, float] = {}
    best_score = -np.inf
    best_genotype = ""
    best_generation = 0

    for generation in range(1, args.generations + 1):
        sequences: list[list[int]] = []
        genotypes: list[str] = []
        completed_flags: list[bool] = []
        while len(sequences) < args.population_size:
            sequence, completed = _sample_sequence(probs, rng, args.max_length)
            if not completed:
                continue
            genotype = _decode_sequence(sequence)
            if not genotype:
                continue
            sequences.append(sequence)
            genotypes.append(genotype)
            completed_flags.append(completed)

        true_scores = np.asarray(_evaluate(genotypes, evaluator, cache), dtype=np.float64)
        objectives = np.asarray([_objective(score, genotype, args) for score, genotype in zip(true_scores, genotypes)], dtype=np.float64)
        order = np.argsort(objectives)[::-1]
        elite_count = max(2, int(args.population_size * args.elite_fraction))
        elite_sequences = [sequences[int(idx)] for idx in order[:elite_count]]
        probs = _update_probs(probs, elite_sequences, args)

        gen_best_idx = int(order[0])
        gen_best_score = float(true_scores[gen_best_idx])
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_genotype = genotypes[gen_best_idx]
            best_generation = generation

        for rank, idx in enumerate(order[: min(10, len(order))], start=1):
            idx = int(idx)
            rows.append(
                {
                    "generation": generation,
                    "rank": rank,
                    "genotype": genotypes[idx],
                    "true_fitness": float(true_scores[idx]),
                    "objective": float(objectives[idx]),
                    "length": len(genotypes[idx]),
                    "completed": completed_flags[idx],
                    "best_true_fitness": best_score,
                    "best_genotype": best_genotype,
                    "evaluated_unique": len(cache),
                }
            )
        history_rows.append(
            {
                "generation": generation,
                "generation_best_true": gen_best_score,
                "generation_best_genotype": genotypes[gen_best_idx],
                "best_true_fitness": best_score,
                "best_generation": best_generation,
                "best_genotype": best_genotype,
                "evaluated_unique": len(cache),
            }
        )
        print(f"Generation {generation}: gen_best={gen_best_score:.6g} best={best_score:.6g} evals={len(cache)}", flush=True)

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = _resolve(f"exports/grammar_cem_{stamp}.csv")
    else:
        output_path = _resolve(args.output)
    history_path = _resolve(args.history_output) if args.history_output else output_path.with_name(f"{output_path.stem}.history.csv")
    details_path = output_path.with_suffix(".details.json")
    plot_path = _resolve(args.plot_output) if args.plot_output else output_path.with_name(f"{output_path.stem}.history.png")
    _write_csv(output_path, rows)
    _write_csv(history_path, history_rows)
    details_path.write_text(json.dumps({"args": vars(args), "best_score": best_score, "best_genotype": best_genotype}, indent=2), encoding="utf-8")
    if not args.no_plot and _plot_history(plot_path, history_rows):
        print(f"Wrote: {plot_path}", flush=True)
    print(f"Wrote: {output_path}", flush=True)
    print(f"Wrote: {history_path}", flush=True)
    print(f"Wrote: {details_path}", flush=True)


if __name__ == "__main__":
    main()
