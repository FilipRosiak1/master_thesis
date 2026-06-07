from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from f1vae.config.io import maybe_resolve_path
from f1vae.data.datasets import GrammarRuleDataset
from f1vae.fitness.framsticks import FramsticksFitness, INVALID_FITNESS
from f1vae.inference.checkpoints import load_checkpoint
from f1vae.models.registry import build_model, dataset_class_for_model

from compare_latent_operators_ea import _frams_ea, _latent_ea
from optimize_selected_latents import (
    _condition_value_for_model,
    _decode_from_z,
    _encode_mu,
    _find_checkpoint,
    _is_valid_f1,
    _load_json,
    _model_config,
    _normalize_genotype,
    _score_genotypes,
    _seeded_cem,
    _seeded_cmaes,
    _tensor_for_genotype,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one final benchmark job from a prepared Slurm plan")
    parser.add_argument("--job-plan", required=True)
    parser.add_argument("--job-index", type=int, default=None, help="0-based row index; defaults to SLURM_ARRAY_TASK_ID")
    parser.add_argument("--seed-manifest", default=None, help="Defaults to seed_manifest.csv next to the job plan")
    parser.add_argument("--require-job-type", choices=("latent", "baseline"), default=None)
    parser.add_argument("--checkpoints-root", default="exports/f1_selected_10_ckpts_20260520_184738")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--framsticks-path", default="src/framsticks/Framsticks54")
    parser.add_argument("--framsticks-lib", default=None)
    parser.add_argument("--framsticks-sim", default="src/framsticks/framspy/eval-allcriteria.sim")
    parser.add_argument("--device", default=None)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--offspring-size", type=int, default=None)
    parser.add_argument("--elite-size", type=int, default=10)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--max-evaluations", type=int, default=None)
    parser.add_argument("--time-budget-seconds", type=float, default=None, help="Optional cap for this single job")
    parser.add_argument("--frams-crossover-prob", type=float, default=0.25)
    parser.add_argument("--frams-mutate-after-crossover-prob", type=float, default=0.25)
    parser.add_argument("--latent-mutation-stds", default="0.12,0.25,0.5,0.9")
    parser.add_argument("--latent-crossover-prob", type=float, default=0.45)
    parser.add_argument("--latent-extrapolate-prob", type=float, default=0.2)
    parser.add_argument("--latent-directional-prob", type=float, default=0.25)
    parser.add_argument("--latent-line-scale", type=float, default=0.6)
    parser.add_argument("--latent-random-immigrant-prob", type=float, default=0.05)
    parser.add_argument("--latent-reencode-offspring", action="store_true")
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--initial-std", type=float, default=0.35)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--min-std", type=float, default=0.02)
    parser.add_argument("--cma-sigma", type=float, default=None)
    parser.add_argument("--condition-fitness", type=float, default=None)
    parser.add_argument("--condition-length", type=float, default=None)
    parser.add_argument("--condition-segments", type=float, default=None)
    parser.add_argument("--attempt-factor", type=int, default=30)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def _resolve_root(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _resolve_plan_path(path: str, plan_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else plan_dir / value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_value(row.get(key)) for key in fieldnames})


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8g}"
    if value is None:
        return ""
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _job_index_from_args(value: int | None) -> int:
    if value is not None:
        return value
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None:
        raise RuntimeError("Missing --job-index and SLURM_ARRAY_TASK_ID")
    return int(raw)


def _parse_float_list(value: str) -> list[float]:
    out = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not out:
        raise ValueError("At least one latent mutation std is required")
    return out


def _load_plan_row(plan_path: Path, index: int) -> dict[str, str]:
    rows = _read_csv(plan_path)
    if index < 0 or index >= len(rows):
        raise IndexError(f"Job index {index} outside plan range 0..{len(rows) - 1}")
    return rows[index]


def _load_seed_rows(seed_manifest: Path, bucket_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _read_csv(seed_manifest):
        if row.get("bucket_id") != bucket_id:
            continue
        rows.append(
            {
                "bucket_id": row["bucket_id"],
                "bucket_min": row.get("bucket_min", ""),
                "bucket_max": row.get("bucket_max", ""),
                "seed_rank": int(row["seed_rank"]),
                "dataset_idx": row.get("dataset_idx", ""),
                "genotype": _normalize_genotype(row["genotype"]),
                "fitness": float(row["fitness"]),
            }
        )
    rows.sort(key=lambda item: int(item["seed_rank"]))
    if not rows:
        raise RuntimeError(f"No seeds found for bucket {bucket_id} in {seed_manifest}")
    return rows


def _known_genotypes_from_dataset(path: Path) -> set[str]:
    known: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            genotype = raw.split("\t", 1)[0] if "\t" in raw else raw.split()[0]
            known.add(_normalize_genotype(genotype))
    return known


def _deadline(args: argparse.Namespace) -> float | None:
    return time.monotonic() + args.time_budget_seconds if args.time_budget_seconds is not None else None


def _common_row(
    *,
    args: argparse.Namespace,
    plan_row: dict[str, str],
    seed_row: dict[str, Any],
    method: str,
    model_label: str,
    model_name: str,
    checkpoint: str,
    seed_decoded: str,
    seed_decoded_score: float | str,
    result: dict[str, Any],
    known_genotypes: set[str],
) -> dict[str, Any]:
    seed_fitness = float(seed_row["fitness"])
    best_score = float(result["best_score"])
    best_genotype = str(result["best_genotype"])
    best_or_source_score = max(best_score, seed_fitness)
    normalized_best = _normalize_genotype(best_genotype)
    normalized_seed = _normalize_genotype(str(seed_row["genotype"]))
    normalized_decoded = _normalize_genotype(seed_decoded) if seed_decoded else ""
    return {
        "global_job_id": plan_row.get("global_job_id", ""),
        "array_job_id": plan_row.get("array_job_id", ""),
        "job_type": plan_row.get("job_type", ""),
        "bucket_id": seed_row["bucket_id"],
        "bucket_min": seed_row.get("bucket_min", ""),
        "bucket_max": seed_row.get("bucket_max", ""),
        "label": model_label,
        "model": model_name,
        "method": method,
        "checkpoint": checkpoint,
        "seed_rank": seed_row["seed_rank"],
        "seed_dataset_idx": seed_row.get("dataset_idx", ""),
        "seed_source": "seed_manifest",
        "seed_source_fitness": seed_fitness,
        "seed_genotype": normalized_seed,
        "seed_decoded_fitness": seed_decoded_score,
        "seed_decoded_genotype": seed_decoded,
        "best_score": best_score,
        "best_or_source_score": best_or_source_score,
        "best_genotype": best_genotype,
        "best_in_dataset": normalized_best in known_genotypes,
        "best_is_source_seed": normalized_best == normalized_seed,
        "best_is_decoded_seed": bool(normalized_decoded) and normalized_best == normalized_decoded,
        "best_minus_seed_source": best_score - seed_fitness,
        "best_or_source_minus_seed_source": best_or_source_score - seed_fitness,
        "best_minus_seed_decoded": _minus(best_score, seed_decoded_score),
        "evaluated_unique": result.get("evaluated_unique", ""),
        "generations_requested": args.generations,
        "iterations_requested": args.iterations,
        "generations_actual": result.get("generations_actual", ""),
        "iterations_actual": result.get("iterations_actual", ""),
        "population_size": args.population_size,
        "offspring_size": args.offspring_size if args.offspring_size is not None else args.population_size,
        "max_evaluations": args.max_evaluations,
        "attempts_total": result.get("attempts_total", ""),
        "duplicate_count": result.get("duplicate_count", ""),
        "invalid_count": result.get("invalid_count", ""),
        "condition_fitness": args.condition_fitness,
        "condition_length": args.condition_length,
        "condition_segments": args.condition_segments,
    }


def _minus(left: float, right: float | str) -> float | str:
    try:
        return left - float(right)
    except (TypeError, ValueError):
        return ""


def _history_rows(
    *,
    base_row: dict[str, Any],
    seed_fitness: float,
    history: list[Any],
    population_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(history, start=1):
        if isinstance(item, dict):
            step = int(item.get("generation", item.get("iteration", idx)))
            score = float(item.get("best_score", INVALID_FITNESS))
            evaluated_unique = item.get("evaluated_unique", "")
            children = item.get("children", "")
            attempts = item.get("attempts", "")
            duplicates = item.get("duplicates", "")
            invalid = item.get("invalid", "")
        else:
            step = idx
            score = float(item)
            evaluated_unique = ""
            children = ""
            attempts = ""
            duplicates = ""
            invalid = ""
        rows.append(
            {
                "global_job_id": base_row.get("global_job_id", ""),
                "array_job_id": base_row.get("array_job_id", ""),
                "job_type": base_row.get("job_type", ""),
                "bucket_id": base_row["bucket_id"],
                "label": base_row.get("label", ""),
                "model": base_row.get("model", ""),
                "method": base_row["method"],
                "seed_rank": base_row["seed_rank"],
                "seed_dataset_idx": base_row.get("seed_dataset_idx", ""),
                "seed_source_fitness": seed_fitness,
                "seed_decoded_fitness": base_row.get("seed_decoded_fitness", ""),
                "step": step,
                "generation": step if base_row["method"] in {"frams", "latent_ea"} else "",
                "iteration": step if base_row["method"] in {"cem", "cmaes"} else "",
                "best_score": score,
                "best_or_source_score": max(score, seed_fitness),
                "best_minus_seed_source": score - seed_fitness,
                "best_or_source_minus_seed_source": max(score, seed_fitness) - seed_fitness,
                "evaluated_unique": evaluated_unique,
                "evaluations_requested": step * population_size,
                "children": children,
                "attempts": attempts,
                "duplicates": duplicates,
                "invalid": invalid,
            }
        )
    return rows


def _run_baseline(
    *,
    args: argparse.Namespace,
    plan_row: dict[str, str],
    seed_rows: list[dict[str, Any]],
    evaluator: FramsticksFitness,
    data_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    known = _known_genotypes_from_dataset(data_path)
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    job_seed = args.seed + int(plan_row.get("global_job_id") or 0) * 1_000_003
    offspring_size = args.offspring_size if args.offspring_size is not None else args.population_size
    deadline = _deadline(args)

    for seed_row in seed_rows:
        if deadline is not None and time.monotonic() >= deadline:
            break
        run_seed = job_seed + int(seed_row["seed_rank"]) * 1009
        print(f"baseline {seed_row['bucket_id']} seed {seed_row['seed_rank']}", flush=True)
        result = _frams_ea(
            seed_genotype=str(seed_row["genotype"]),
            seed_fitness=float(seed_row["fitness"]),
            evaluator=evaluator,
            rng=random.Random(run_seed),
            generations=args.generations,
            population_size=args.population_size,
            offspring_size=offspring_size,
            elite_size=args.elite_size,
            tournament_size=args.tournament_size,
            max_evaluations=args.max_evaluations,
            deadline=deadline,
            crossover_prob=args.frams_crossover_prob,
            mutate_after_crossover_prob=args.frams_mutate_after_crossover_prob,
            attempt_factor=args.attempt_factor,
        )
        result["generations_actual"] = len(result.get("history", []))
        row = _common_row(
            args=args,
            plan_row=plan_row,
            seed_row=seed_row,
            method="frams",
            model_label="",
            model_name="",
            checkpoint="",
            seed_decoded="",
            seed_decoded_score="",
            result=result,
            known_genotypes=known,
        )
        rows.append(row)
        histories.extend(_history_rows(base_row=row, seed_fitness=float(seed_row["fitness"]), history=result.get("history", []), population_size=args.population_size))
        details.append({**row, "history": result.get("history", []), "final_population": result.get("final_population", [])})
        print(f"  best={float(result['best_score']):.6g} delta={float(result['best_score']) - float(seed_row['fitness']):.6g}", flush=True)
    return rows, histories, details


def _load_latent_context(args: argparse.Namespace, label: str, evaluator: FramsticksFitness) -> dict[str, Any]:
    checkpoints_root = Path(maybe_resolve_path(args.checkpoints_root, root_dir=str(ROOT)))
    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = checkpoints_root / label
    if not run_dir.exists():
        label_path = Path(label)
        if label_path.is_absolute() and label_path.exists():
            run_dir = label_path
        elif (ROOT / label_path).exists():
            run_dir = ROOT / label_path
    checkpoint_path = _find_checkpoint(run_dir)
    run_config = _load_json(run_dir / "run_config.json")
    state_dict, checkpoint_meta = load_checkpoint(str(checkpoint_path), device)
    cfg = _model_config(run_config, checkpoint_meta)
    model_name = cfg["model_name"]
    dataset_cls = dataset_class_for_model(model_name)
    dataset = dataset_cls(data_path, cfg["max_length"])
    if not isinstance(dataset, GrammarRuleDataset):
        raise TypeError(f"Only grammar-rule datasets are supported, got {type(dataset).__name__}")
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
    evaluator.evaluate_one("X")
    return {
        "checkpoints_root": checkpoints_root,
        "run_dir": run_dir,
        "checkpoint_path": checkpoint_path,
        "run_config": run_config,
        "checkpoint_meta": checkpoint_meta,
        "cfg": cfg,
        "model_name": model_name,
        "dataset": dataset,
        "model": model,
        "device": device,
        "condition_value": _condition_value_for_model(model, args, run_config, checkpoint_meta),
        "known_genotypes": {_normalize_genotype(genotype) for genotype in dataset.valid_lines},
    }


def _run_latent(
    *,
    args: argparse.Namespace,
    plan_row: dict[str, str],
    seed_rows: list[dict[str, Any]],
    evaluator: FramsticksFitness,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    label = plan_row["model_label"]
    method = plan_row["method"]
    if method not in {"latent_ea", "cem", "cmaes"}:
        raise ValueError(f"Unsupported latent method: {method}")
    context = _load_latent_context(args, label, evaluator)
    cfg = context["cfg"]
    model_name = context["model_name"]
    model = context["model"]
    device = context["device"]
    condition_value = context["condition_value"]
    checkpoint = context["checkpoint_path"].relative_to(context["run_dir"]).as_posix()
    mutation_stds = _parse_float_list(args.latent_mutation_stds)
    rows: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    job_seed = args.seed + int(plan_row.get("global_job_id") or 0) * 1_000_003
    offspring_size = args.offspring_size if args.offspring_size is not None else args.population_size
    deadline = _deadline(args)

    with torch.no_grad():
        for seed_row in seed_rows:
            if deadline is not None and time.monotonic() >= deadline:
                break
            seed_genotype = str(seed_row["genotype"])
            tensor = _tensor_for_genotype(seed_genotype, cfg["max_length"])
            if tensor is None:
                raise ValueError(f"Seed cannot be encoded by current grammar: {seed_genotype}")
            seed_z = _encode_mu(model_name, model, tensor.to(device)).squeeze(0).detach().cpu().numpy()
            score_cache: dict[str, float] = {}
            seed_decoded = _decode_from_z(model_name, model, seed_z, device, condition_value)
            seed_decoded_score = _score_genotypes([seed_decoded], evaluator, score_cache)[0]
            run_seed = job_seed + int(seed_row["seed_rank"]) * 1009
            print(
                f"{method} {label} {seed_row['bucket_id']} seed {seed_row['seed_rank']}: "
                f"source={float(seed_row['fitness']):.6g} decoded={seed_decoded_score:.6g}",
                flush=True,
            )

            if method == "latent_ea":
                result = _latent_ea(
                    model_name=model_name,
                    model=model,
                    seed_z=seed_z,
                    seed_genotype=seed_genotype,
                    seed_fitness=float(seed_row["fitness"]),
                    evaluator=evaluator,
                    device=device,
                    rng=random.Random(run_seed),
                    rng_np=np.random.default_rng(run_seed),
                    generations=args.generations,
                    population_size=args.population_size,
                    offspring_size=offspring_size,
                    elite_size=args.elite_size,
                    tournament_size=args.tournament_size,
                    max_evaluations=args.max_evaluations,
                    deadline=deadline,
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
                result["generations_actual"] = len(result.get("history", []))
            elif method == "cem":
                result = _seeded_cem(
                    model_name=model_name,
                    model=model,
                    seed_z=seed_z,
                    evaluator=evaluator,
                    device=device,
                    rng=np.random.default_rng(run_seed),
                    iterations=args.iterations,
                    population_size=args.population_size,
                    elite_fraction=args.elite_fraction,
                    initial_std=args.initial_std,
                    smoothing=args.smoothing,
                    min_std=args.min_std,
                    score_cache=score_cache,
                    deadline=deadline,
                    condition_value=condition_value,
                )
                result["evaluated_unique"] = len(score_cache)
                result["iterations_actual"] = len(result.get("history", []))
            else:
                result = _seeded_cmaes(
                    model_name=model_name,
                    model=model,
                    seed_z=seed_z,
                    evaluator=evaluator,
                    device=device,
                    rng=np.random.default_rng(run_seed),
                    iterations=args.iterations,
                    population_size=args.population_size,
                    sigma=args.cma_sigma if args.cma_sigma is not None else args.initial_std,
                    score_cache=score_cache,
                    deadline=deadline,
                    condition_value=condition_value,
                )
                result["evaluated_unique"] = len(score_cache)
                result["iterations_actual"] = len(result.get("history", []))

            row = _common_row(
                args=args,
                plan_row=plan_row,
                seed_row=seed_row,
                method=method,
                model_label=label,
                model_name=model_name,
                checkpoint=checkpoint,
                seed_decoded=seed_decoded,
                seed_decoded_score=seed_decoded_score,
                result=result,
                known_genotypes=context["known_genotypes"],
            )
            rows.append(row)
            histories.extend(_history_rows(base_row=row, seed_fitness=float(seed_row["fitness"]), history=result.get("history", []), population_size=args.population_size))
            detail = {**row, "history": result.get("history", []), "best_z": result.get("best_z", "")}
            if method == "latent_ea":
                detail["final_population"] = result.get("final_population", [])
            details.append(detail)
            print(f"  best={float(result['best_score']):.6g} delta={float(result['best_score']) - float(seed_row['fitness']):.6g}", flush=True)
    return rows, histories, details


def main() -> None:
    args = _build_parser().parse_args()
    plan_path = _resolve_root(args.job_plan)
    plan_dir = plan_path.parent
    job_index = _job_index_from_args(args.job_index)
    plan_row = _load_plan_row(plan_path, job_index)
    if args.require_job_type and plan_row.get("job_type") != args.require_job_type:
        raise RuntimeError(f"Plan row job_type={plan_row.get('job_type')} does not match required {args.require_job_type}")

    seed_manifest = _resolve_root(args.seed_manifest) if args.seed_manifest else plan_dir / "seed_manifest.csv"
    seed_rows = _load_seed_rows(seed_manifest, plan_row["bucket_id"])
    output_csv = _resolve_plan_path(plan_row["output_csv"], plan_dir)
    output_history = _resolve_plan_path(plan_row["output_history"], plan_dir)
    output_details = _resolve_plan_path(plan_row["output_details"], plan_dir)
    if args.skip_existing and output_csv.exists() and output_history.exists() and output_details.exists():
        print(f"Skipping existing job output: {output_csv}", flush=True)
        return

    data_path = Path(maybe_resolve_path(args.data_path, root_dir=str(ROOT)))
    evaluator = FramsticksFitness(
        maybe_resolve_path(args.framsticks_path, root_dir=str(ROOT)),
        lib=args.framsticks_lib,
        sim=maybe_resolve_path(args.framsticks_sim, root_dir=str(ROOT)),
    )
    evaluator.evaluate_one("X")

    print(
        f"Running job index {job_index}: type={plan_row['job_type']} method={plan_row['method']} "
        f"label={plan_row.get('model_label', '')} bucket={plan_row['bucket_id']} seeds={len(seed_rows)}",
        flush=True,
    )
    if plan_row["job_type"] == "baseline":
        rows, history_rows, run_details = _run_baseline(
            args=args,
            plan_row=plan_row,
            seed_rows=seed_rows,
            evaluator=evaluator,
            data_path=data_path,
        )
    elif plan_row["job_type"] == "latent":
        rows, history_rows, run_details = _run_latent(
            args=args,
            plan_row=plan_row,
            seed_rows=seed_rows,
            evaluator=evaluator,
        )
    else:
        raise ValueError(f"Unsupported job_type: {plan_row['job_type']}")

    _write_csv(output_csv, rows)
    _write_csv(output_history, history_rows)
    details = {
        "job_index": job_index,
        "plan_path": str(plan_path),
        "seed_manifest": str(seed_manifest),
        "plan_row": plan_row,
        "args": vars(args),
        "runs": run_details,
    }
    output_details.parent.mkdir(parents=True, exist_ok=True)
    output_details.write_text(json.dumps(details, indent=2, default=_json_default), encoding="utf-8")
    print(f"Wrote: {output_csv}", flush=True)
    print(f"Wrote: {output_history}", flush=True)
    print(f"Wrote: {output_details}", flush=True)


if __name__ == "__main__":
    main()
