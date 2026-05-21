from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ALL_LABELS = (
    "01_lhs_standard_null_best",
    "02_lhs_latent256_seed42_best",
    "03_masked_standard_null_best",
    "04_lhs_standard_seed42_best",
    "05_masked_slow_null_best",
    "06_lhs_depth_seed42_best",
    "07_lhs_slow_null_best",
    "08_masked_reconstruction_ceiling_best",
    "09_tree_vae_epoch80",
    "10_transformer_vae_epoch90",
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark selected latent optimizers with capped starting seeds")
    parser.add_argument("--checkpoints-root", default="exports/f1_selected_10_ckpts_20260520_184738")
    parser.add_argument("--labels", default=",".join(ALL_LABELS), help="Comma-separated checkpoint folder labels")
    parser.add_argument("--algorithms", default="cem,cmaes", help="Comma-separated subset of: cem,cmaes")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--framsticks-path", default="src/framsticks/Framsticks54")
    parser.add_argument("--framsticks-lib", default=None)
    parser.add_argument("--framsticks-sim", default="src/framsticks/framspy/eval-allcriteria.sim")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed-candidates", type=int, default=5)
    parser.add_argument("--seed-source", choices=("dataset_top", "simplest", "framsticks_mutations"), default="dataset_top")
    parser.add_argument("--seed-min-fitness", type=float, default=None)
    parser.add_argument("--seed-max-fitness", type=float, default=1.2)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--time-budget-seconds", type=float, default=600.0, help="Per-label optimizer budget")
    parser.add_argument("--split-time-budget-across-seeds", dest="split_time_budget_across_seeds", action="store_true", default=True)
    parser.add_argument("--no-split-time-budget-across-seeds", dest="split_time_budget_across_seeds", action="store_false")
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--initial-std", type=float, default=0.35)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--min-std", type=float, default=0.02)
    parser.add_argument("--mutation-pool-size", type=int, default=200)
    parser.add_argument("--mutation-attempts", type=int, default=2000)
    parser.add_argument("--cma-sigma", type=float, default=None)
    parser.add_argument("--condition-fitness", type=float, default=None, help="Raw target fitness for conditional decoders")
    parser.add_argument("--condition-length", type=float, default=None, help="Raw target genotype string length for structural conditional decoders")
    parser.add_argument("--condition-segments", type=float, default=None, help="Raw target X segment count for structural conditional decoders")
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--output", default=None, help="Combined CSV output path")
    parser.add_argument("--history-output", default=None, help="Combined per-iteration history CSV path")
    parser.add_argument("--plot-output", default=None, help="Per-iteration progress plot PNG path")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolved(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _add_optional(cmd: list[str], flag: str, value: Any) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _algorithm_output_path(combined_path: Path, algorithm: str) -> Path:
    return combined_path.with_name(f"{combined_path.stem}.{algorithm}{combined_path.suffix}")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_combined(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _read_details(path: Path) -> dict[str, Any]:
    details_path = path.with_suffix(".details.json")
    if not details_path.exists():
        return {}
    return json.loads(details_path.read_text(encoding="utf-8"))


def _history_rows(details: dict[str, Any], algorithm: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, runs in details.get("runs", {}).items():
        for run in runs:
            seed_source_fitness = _safe_float(run.get("seed_source_fitness"))
            seed_decoded_fitness = _safe_float(run.get("seed_decoded_fitness"))
            for iteration, score in enumerate(run.get("history", []), start=1):
                score_value = _safe_float(score)
                rows.append(
                    {
                        "algorithm": algorithm,
                        "label": label,
                        "model": run.get("model", ""),
                        "seed_rank": run.get("seed_rank", ""),
                        "seed_source": run.get("seed_source", ""),
                        "seed_source_fitness": seed_source_fitness,
                        "seed_decoded_fitness": seed_decoded_fitness,
                        "iteration": iteration,
                        "best_score": score_value,
                        "best_minus_seed_source": _minus(score_value, seed_source_fitness),
                        "best_minus_seed_decoded": _minus(score_value, seed_decoded_fitness),
                    }
                )
    return rows


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _minus(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _format_cell(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.8g}"
    if value is None:
        return ""
    return value


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_cell(row.get(key)) for key in fieldnames})


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
        key = (str(row.get("algorithm", "")), str(row.get("label", "")), str(row.get("seed_rank", "")))
        grouped.setdefault(key, []).append(row)

    fig, ax = plt.subplots(figsize=(14, 8))
    for (algorithm, label, seed_rank), group in sorted(grouped.items()):
        group.sort(key=lambda item: int(item["iteration"]))
        x_values = [int(item["iteration"]) for item in group]
        y_values = [float(item["best_score"]) for item in group if item.get("best_score") is not None]
        if len(y_values) != len(x_values):
            continue
        ax.plot(x_values, y_values, linewidth=1.2, label=f"{label} {algorithm} s{seed_rank}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best vertpos so far")
    ax.set_title("Latent optimization progress")
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


def main() -> None:
    args = _build_parser().parse_args()
    algorithms = _split_csv(args.algorithms)
    labels = _split_csv(args.labels)
    invalid = sorted(set(algorithms) - {"cem", "cmaes"})
    if invalid:
        raise ValueError(f"Unsupported algorithms: {', '.join(invalid)}")
    if not algorithms:
        raise ValueError("At least one algorithm is required")
    if not labels:
        raise ValueError("At least one label is required")

    checkpoints_root = _resolved(args.checkpoints_root)
    if args.output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = checkpoints_root / f"latent_optimization_all10_capped_{stamp}.csv"
    else:
        combined_path = _resolved(args.output)
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    optimizer = ROOT / "scripts" / "optimize_selected_latents.py"
    all_rows: list[dict[str, str]] = []
    details: dict[str, Any] = {
        "labels": labels,
        "algorithms": algorithms,
        "seed_source": args.seed_source,
        "seed_min_fitness": args.seed_min_fitness,
        "seed_max_fitness": args.seed_max_fitness,
        "time_budget_seconds": args.time_budget_seconds,
        "split_time_budget_across_seeds": args.split_time_budget_across_seeds,
        "condition_fitness": args.condition_fitness,
        "condition_length": args.condition_length,
        "condition_segments": args.condition_segments,
        "runs": {},
    }
    all_history_rows: list[dict[str, Any]] = []

    for algorithm in algorithms:
        algorithm_output = _algorithm_output_path(combined_path, algorithm)
        cmd = [
            args.python,
            str(optimizer),
            "--checkpoints-root",
            str(checkpoints_root),
            "--labels",
            ",".join(labels),
            "--data-path",
            str(_resolved(args.data_path)),
            "--framsticks-path",
            str(_resolved(args.framsticks_path)),
            "--framsticks-sim",
            str(_resolved(args.framsticks_sim)),
            "--seed-candidates",
            str(args.seed_candidates),
            "--seed-source",
            args.seed_source,
            "--algorithm",
            algorithm,
            "--iterations",
            str(args.iterations),
            "--time-budget-seconds",
            str(args.time_budget_seconds),
            "--population-size",
            str(args.population_size),
            "--elite-fraction",
            str(args.elite_fraction),
            "--initial-std",
            str(args.initial_std),
            "--smoothing",
            str(args.smoothing),
            "--min-std",
            str(args.min_std),
            "--mutation-pool-size",
            str(args.mutation_pool_size),
            "--mutation-attempts",
            str(args.mutation_attempts),
            "--seed",
            str(args.seed),
            "--output",
            str(algorithm_output),
        ]
        _add_optional(cmd, "--framsticks-lib", args.framsticks_lib)
        _add_optional(cmd, "--device", args.device)
        _add_optional(cmd, "--seed-min-fitness", args.seed_min_fitness)
        _add_optional(cmd, "--seed-max-fitness", args.seed_max_fitness)
        _add_optional(cmd, "--cma-sigma", args.cma_sigma)
        _add_optional(cmd, "--condition-fitness", args.condition_fitness)
        _add_optional(cmd, "--condition-length", args.condition_length)
        _add_optional(cmd, "--condition-segments", args.condition_segments)
        if args.split_time_budget_across_seeds:
            cmd.append("--split-time-budget-across-seeds")

        print("Running:", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=ROOT, check=True)
        rows = _read_rows(algorithm_output)
        all_rows.extend(rows)
        algorithm_details = _read_details(algorithm_output)
        all_history_rows.extend(_history_rows(algorithm_details, algorithm))
        details["runs"][algorithm] = {
            "csv": str(algorithm_output),
            "details": algorithm_details,
        }

    if args.dry_run:
        return

    _write_combined(combined_path, all_rows)
    history_path = _resolved(args.history_output) if args.history_output is not None else combined_path.with_name(f"{combined_path.stem}.history.csv")
    _write_history(history_path, all_history_rows)
    if not args.no_plot:
        plot_path = _resolved(args.plot_output) if args.plot_output is not None else combined_path.with_name(f"{combined_path.stem}.history.png")
        if _plot_history(plot_path, all_history_rows):
            print(f"Wrote: {plot_path}")
    details_path = combined_path.with_suffix(".details.json")
    details_path.write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"Wrote: {combined_path}")
    print(f"Wrote: {history_path}")
    print(f"Wrote: {details_path}")


if __name__ == "__main__":
    main()
