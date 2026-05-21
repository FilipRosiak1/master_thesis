from __future__ import annotations

import argparse
import csv
import json
import math
import random
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

from f1vae.config.defaults import DEFAULTS
from f1vae.config.io import maybe_resolve_path
from f1vae.data.datasets import GrammarRuleDataset
from f1vae.fitness.framsticks import FramsticksFitness, INVALID_FITNESS
from f1vae.inference.checkpoints import load_checkpoint
from f1vae.models.registry import build_model, dataset_class_for_model

from optimize_selected_latents import (
    TREE_MODELS,
    _decode_from_z,
    _decode_many,
    _encode_mu,
    _fitness_in_range,
    _is_valid_f1,
    _make_seed_entries,
    _score_genotypes,
    _seeded_cem,
    _seeded_cmaes,
    _tensor_for_genotype,
)


DEFAULT_LABELS = "09_tree_vae_epoch80,10_transformer_vae_epoch90,08_masked_reconstruction_ceiling_best"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Framsticks mutations against latent-space search from matched seeds")
    parser.add_argument("--checkpoints-root", default="exports/f1_selected_10_ckpts_20260520_184738")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--methods", default="framsticks,latent_random,cem,cmaes")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--framsticks-path", default="src/framsticks/Framsticks54")
    parser.add_argument("--framsticks-lib", default=None)
    parser.add_argument("--framsticks-sim", default="src/framsticks/framspy/eval-allcriteria.sim")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed-source", choices=("dataset_top", "simplest", "framsticks_mutations"), default="dataset_top")
    parser.add_argument("--seed-candidates", type=int, default=5)
    parser.add_argument("--seed-min-fitness", type=float, default=None)
    parser.add_argument("--seed-max-fitness", type=float, default=1.2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--latent-std", type=float, default=0.35)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--min-std", type=float, default=0.02)
    parser.add_argument("--cma-sigma", type=float, default=None)
    parser.add_argument("--mutation-pool-size", type=int, default=200)
    parser.add_argument("--mutation-attempts", type=int, default=2000)
    parser.add_argument("--mutation-require-grammar", action="store_true")
    parser.add_argument("--condition-fitness", type=float, default=None)
    parser.add_argument("--condition-length", type=float, default=None)
    parser.add_argument("--condition-segments", type=float, default=None)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output", default=None)
    parser.add_argument("--history-output", default=None)
    parser.add_argument("--plot-output", default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_checkpoint(run_dir: Path) -> Path:
    best = run_dir / "best_val_recon.pth"
    if best.exists():
        return best
    checkpoints = sorted(run_dir.rglob("*.pth"), key=lambda item: item.as_posix())
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


def _normalize_genotype(genotype: str) -> str:
    return "".join(genotype.split())


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


def _score_raw_framsticks(
    genotypes: list[str],
    evaluator: FramsticksFitness,
    cache: dict[str, float],
) -> list[float]:
    to_eval: list[str] = []
    for genotype in genotypes:
        normalized = _normalize_genotype(genotype)
        if normalized not in cache and normalized not in to_eval:
            to_eval.append(normalized)
    if to_eval:
        values = evaluator.evaluate_many(to_eval)
        if len(values) != len(to_eval):
            values = [evaluator.evaluate_one(genotype) for genotype in to_eval]
        for genotype, value in zip(to_eval, values):
            cache[genotype] = INVALID_FITNESS if value is None else float(value)
    return [cache[_normalize_genotype(genotype)] for genotype in genotypes]


def _framsticks_mutation_search(
    *,
    seed_genotype: str,
    seed_score: float,
    evaluator: FramsticksFitness,
    py_rng: random.Random,
    iterations: int,
    population_size: int,
    elite_fraction: float,
    max_length: int,
    require_grammar: bool,
    score_cache: dict[str, float],
) -> dict[str, Any]:
    frams_lib = evaluator._ensure_loaded()
    elite_count = max(1, int(population_size * elite_fraction))
    parents = [_normalize_genotype(seed_genotype)]
    best_genotype = parents[0]
    best_score = seed_score
    score_cache[best_genotype] = seed_score
    history: list[float] = []

    for _ in range(iterations):
        children: list[str] = []
        attempts = 0
        while len(children) < population_size and attempts < population_size * 20:
            attempts += 1
            parent = py_rng.choice(parents)
            try:
                child = _normalize_genotype(frams_lib.mutate([parent])[0])
            except Exception:
                continue
            if require_grammar and (not _is_valid_f1(child) or _tensor_for_genotype(child, max_length) is None):
                continue
            children.append(child)

        if not children:
            history.append(best_score)
            continue

        scores = _score_raw_framsticks(children, evaluator, score_cache)
        for genotype, score in zip(children, scores):
            if score > best_score:
                best_score = score
                best_genotype = genotype
        combined = list({*parents, *children})
        combined_scores = _score_raw_framsticks(combined, evaluator, score_cache)
        order = np.argsort(np.asarray(combined_scores, dtype=np.float64))[::-1]
        parents = [combined[int(idx)] for idx in order[:elite_count]]
        history.append(best_score)

    return {"best_score": best_score, "best_genotype": best_genotype, "history": history, "best_z": []}


def _latent_random_search(
    *,
    model_name: str,
    model,
    seed_z: np.ndarray,
    evaluator: FramsticksFitness,
    device: torch.device,
    rng: np.random.Generator,
    iterations: int,
    population_size: int,
    latent_std: float,
    score_cache: dict[str, float],
    condition_value: float | None,
) -> dict[str, Any]:
    best_score = -np.inf
    best_genotype = ""
    best_z = seed_z.astype(np.float64).copy()
    history: list[float] = []
    for _ in range(iterations):
        population = rng.normal(seed_z, latent_std, size=(population_size, seed_z.size))
        population[0] = seed_z
        genotypes = _decode_many(model_name, model, population, device, condition_value)
        scores = np.asarray(_score_genotypes(genotypes, evaluator, score_cache), dtype=np.float64)
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) > best_score:
            best_score = float(scores[best_idx])
            best_genotype = genotypes[best_idx]
            best_z = population[best_idx].copy()
        history.append(best_score)
    return {"best_score": best_score, "best_genotype": best_genotype, "history": history, "best_z": best_z.tolist()}


def _plot_history(path: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib is unavailable; skipped progress plot", flush=True)
        return False

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["label"]), str(row["method"]), str(row["seed_rank"]))
        grouped.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(14, 8))
    for (label, method, seed_rank), group in sorted(grouped.items()):
        group.sort(key=lambda item: int(item["iteration"]))
        ax.plot(
            [int(item["iteration"]) for item in group],
            [float(item["best_score"]) for item in group],
            linewidth=1.2,
            label=f"{label} {method} s{seed_rank}",
        )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best vertpos so far")
    ax.set_title("Matched seeded search comparison")
    ax.grid(True, alpha=0.25)
    if len(grouped) <= 30:
        ax.legend(fontsize=6, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout(rect=(0, 0, 0.78, 1))
    else:
        fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _normalize_target(raw: float | None, mean: Any, std: Any, default: float = 0.0) -> float:
    if raw is None:
        return default
    if mean is None or std is None:
        return float(raw)
    return (float(raw) - float(mean)) / max(float(std), 1e-8)


def _condition_value(model, args, run_config: dict[str, Any], checkpoint_meta: dict[str, Any]) -> Any:
    condition_dim = int(getattr(model, "condition_dim", 0))
    if condition_dim <= 0:
        return None
    fitness = _normalize_target(
        args.condition_fitness,
        checkpoint_meta.get("fitness_mean", run_config.get("fitness_mean")),
        checkpoint_meta.get("fitness_std", run_config.get("fitness_std")),
    )
    if condition_dim == 1:
        return fitness if args.condition_fitness is not None else None
    length = _normalize_target(
        args.condition_length,
        checkpoint_meta.get("length_mean", run_config.get("length_mean")),
        checkpoint_meta.get("length_std", run_config.get("length_std")),
    )
    segments = _normalize_target(
        args.condition_segments,
        checkpoint_meta.get("segments_mean", run_config.get("segments_mean")),
        checkpoint_meta.get("segments_std", run_config.get("segments_std")),
    )
    return [fitness, length, segments][:condition_dim]


def main() -> None:
    args = _build_parser().parse_args()
    labels = _split_csv(args.labels)
    methods = _split_csv(args.methods)
    allowed = {"framsticks", "latent_random", "cem", "cmaes"}
    invalid = sorted(set(methods) - allowed)
    if invalid:
        raise ValueError(f"Unsupported methods: {', '.join(invalid)}")

    checkpoints_root = _resolve(args.checkpoints_root)
    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    evaluator = FramsticksFitness(
        maybe_resolve_path(args.framsticks_path, root_dir=str(ROOT)),
        lib=args.framsticks_lib,
        sim=maybe_resolve_path(args.framsticks_sim, root_dir=str(ROOT)),
    )
    evaluator.evaluate_one("X")

    rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"runs": {}, "methods": methods, "labels": labels, "args": vars(args)}

    for label in labels:
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
        condition_value = _condition_value(model, args, run_config, checkpoint_meta)

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

        label_details: list[dict[str, Any]] = []
        with torch.no_grad():
            for seed_entry in seed_entries:
                seed_rank = seed_entry["rank"]
                seed_genotype = seed_entry["genotype"]
                seed_source_fitness = float(seed_entry["fitness"])
                batch = seed_entry["tensor"].to(device)
                seed_z = _encode_mu(model_name, model, batch).squeeze(0).detach().cpu().numpy()
                seed_decoded = _decode_from_z(model_name, model, seed_z, device, condition_value)
                seed_decoded_cache: dict[str, float] = {}
                seed_decoded_fitness = _score_genotypes([seed_decoded], evaluator, seed_decoded_cache)[0]

                for method in methods:
                    print(f"  {label} seed {seed_rank} method {method}", flush=True)
                    score_cache: dict[str, float] = {}
                    if method == "framsticks":
                        result = _framsticks_mutation_search(
                            seed_genotype=seed_genotype,
                            seed_score=seed_source_fitness,
                            evaluator=evaluator,
                            py_rng=py_rng,
                            iterations=args.iterations,
                            population_size=args.population_size,
                            elite_fraction=args.elite_fraction,
                            max_length=cfg["max_length"],
                            require_grammar=args.mutation_require_grammar,
                            score_cache=score_cache,
                        )
                        decoded_reference = seed_source_fitness
                    elif method == "latent_random":
                        result = _latent_random_search(
                            model_name=model_name,
                            model=model,
                            seed_z=seed_z,
                            evaluator=evaluator,
                            device=device,
                            rng=rng,
                            iterations=args.iterations,
                            population_size=args.population_size,
                            latent_std=args.latent_std,
                            score_cache=score_cache,
                            condition_value=condition_value,
                        )
                        decoded_reference = seed_decoded_fitness
                    elif method == "cem":
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
                            initial_std=args.latent_std,
                            smoothing=args.smoothing,
                            min_std=args.min_std,
                            score_cache=score_cache,
                            deadline=None,
                            condition_value=condition_value,
                        )
                        decoded_reference = seed_decoded_fitness
                    else:
                        result = _seeded_cmaes(
                            model_name=model_name,
                            model=model,
                            seed_z=seed_z,
                            evaluator=evaluator,
                            device=device,
                            rng=rng,
                            iterations=args.iterations,
                            population_size=args.population_size,
                            sigma=args.cma_sigma if args.cma_sigma is not None else args.latent_std,
                            score_cache=score_cache,
                            deadline=None,
                            condition_value=condition_value,
                        )
                        decoded_reference = seed_decoded_fitness

                    best_score = float(result["best_score"])
                    best_genotype = str(result["best_genotype"])
                    best_normalized = _normalize_genotype(best_genotype)
                    row = {
                        "label": label,
                        "model": model_name,
                        "method": method,
                        "checkpoint": checkpoint_path.relative_to(run_dir).as_posix(),
                        "seed_rank": seed_rank,
                        "seed_source": args.seed_source,
                        "seed_source_fitness": seed_source_fitness,
                        "seed_decoded_fitness": seed_decoded_fitness if method != "framsticks" else "",
                        "seed_genotype": seed_genotype,
                        "seed_decoded_genotype": seed_decoded if method != "framsticks" else "",
                        "best_score": best_score,
                        "best_genotype": best_genotype,
                        "best_in_dataset": best_normalized in known_genotypes,
                        "best_is_source_seed": best_normalized == _normalize_genotype(seed_genotype),
                        "best_is_decoded_seed": best_normalized == _normalize_genotype(seed_decoded) if method != "framsticks" else "",
                        "best_minus_seed_source": best_score - seed_source_fitness,
                        "best_minus_seed_decoded": best_score - decoded_reference,
                        "evaluated_unique": len(score_cache),
                        "iterations": args.iterations,
                        "population_size": args.population_size,
                        "seed_min_fitness": args.seed_min_fitness,
                        "seed_max_fitness": args.seed_max_fitness,
                        "condition_fitness": args.condition_fitness,
                        "condition_length": args.condition_length,
                        "condition_segments": args.condition_segments,
                        "condition_value": condition_value,
                        "mutation_require_grammar": args.mutation_require_grammar,
                    }
                    rows.append(row)
                    label_details.append({**row, "history": result["history"], "best_z": result.get("best_z", [])})
                    for iteration, score in enumerate(result["history"], start=1):
                        history_rows.append(
                            {
                                "label": label,
                                "model": model_name,
                                "method": method,
                                "seed_rank": seed_rank,
                                "iteration": iteration,
                                "best_score": float(score),
                                "seed_source_fitness": seed_source_fitness,
                                "seed_decoded_fitness": seed_decoded_fitness if method != "framsticks" else "",
                                "best_minus_seed_source": float(score) - seed_source_fitness,
                                "best_minus_seed_decoded": float(score) - decoded_reference,
                            }
                        )
                    print(f"    best={best_score:.6g} delta_source={best_score - seed_source_fitness:.6g}", flush=True)
        details["runs"][label] = label_details

    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = checkpoints_root / f"seeded_method_comparison_{stamp}.csv"
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
