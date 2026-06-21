from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FITNESS_AWARE_LABEL = "exports/cluster_export_20260524_222538/models/f1_guided/fitness_aware_lhs_seed42"
LABEL_NAMES = {
    "09_tree_vae_epoch80": "TreeVAE epoch80",
    "10_transformer_vae_epoch90": "TransformerVAE epoch90",
    "02_lhs_latent256_seed42_best": "LHS latent256",
    "06_lhs_depth_seed42_best": "LHS depth",
    FITNESS_AWARE_LABEL: "Fitness-aware LHS",
    "": "Framsticks EA",
}
METHOD_NAMES = {
    "frams": "Framsticks EA",
    "latent_ea": "EA in hidden space",
    "cem": "CEM",
    "cmaes": "CMA-ES",
}
BUCKET_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6"]
BUCKET_LABELS = {
    "B1": "0.30-0.50",
    "B2": "0.50-0.75",
    "B3": "0.75-1.00",
    "B4": "1.00-1.20",
    "B5": "1.20-1.50",
    "B6": ">=1.50",
}


def _build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Plot best-so-far trajectories for the final benchmark")
    parser.add_argument(
        "--cpu-run",
        default="exports/final_results_cpu/exports/final_seed_bucket_benchmark_baseline_20260613_011516",
    )
    parser.add_argument(
        "--gpu-run",
        default="exports/final_results_gpu/exports/final_seed_bucket_benchmark_20260607_194941",
    )
    parser.add_argument("--output-dir", default=f"exports/final_benchmark_history_plots_{stamp}")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_rows(run_dir: Path, subdir: str, *, history: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    suffix = ".history.csv" if history else ".csv"
    for path in sorted((run_dir / "jobs" / subdir).glob(f"*{suffix}")):
        if not history and path.name.endswith(".history.csv"):
            continue
        rows.extend(_read_csv(path))
    return rows


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _seed_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("bucket_id", ""), row.get("seed_rank", ""), row.get("seed_dataset_idx", ""))


def _history_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("label", ""),
        row.get("method", ""),
        row.get("bucket_id", ""),
        row.get("seed_rank", ""),
        row.get("seed_dataset_idx", ""),
        row.get("model", ""),
    )


def _summary_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("label", ""),
        row.get("method", ""),
        row.get("bucket_id", ""),
        row.get("seed_rank", ""),
        row.get("seed_dataset_idx", ""),
        row.get("model", ""),
    )


def _label_name(label: str) -> str:
    if label in LABEL_NAMES:
        return LABEL_NAMES[label]
    return Path(label.replace("\\", "/")).name if "/" in label or "\\" in label else label


def _method_name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def _median(values: list[float]) -> float:
    values = sorted(value for value in values if not math.isnan(value))
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def _mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else float("nan")


def _quantile(values: list[float], q: float) -> float:
    values = sorted(value for value in values if not math.isnan(value))
    if not values:
        return float("nan")
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - idx) + values[hi] * (idx - lo)


def _setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "#0f172a",
            "axes.facecolor": "#0f172a",
            "savefig.facecolor": "#0f172a",
            "axes.edgecolor": "#94a3b8",
            "axes.labelcolor": "#e5e7eb",
            "xtick.color": "#e5e7eb",
            "ytick.color": "#e5e7eb",
            "text.color": "#f8fafc",
            "axes.titlecolor": "#f8fafc",
            "font.size": 10,
            "axes.grid": True,
            "grid.color": "#334155",
            "grid.alpha": 0.35,
        }
    )
    return plt


def _save(fig, output_dir: Path, filename: str, dpi: int) -> Path:
    path = output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def _x_value(row: dict[str, str]) -> int:
    value = _i(row.get("evaluations_requested"), default=0)
    if value > 0:
        return value
    return max(1, _i(row.get("step"), default=0)) * 100


def _aggregate_curve(rows: list[dict[str, str]], *, value_col: str = "best_score") -> tuple[list[int], list[float], list[float], list[float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[_x_value(row)].append(_f(row.get(value_col)))
    xs = sorted(grouped)
    med = [_median(grouped[x]) for x in xs]
    q25 = [_quantile(grouped[x], 0.25) for x in xs]
    q75 = [_quantile(grouped[x], 0.75) for x in xs]
    return xs, med, q25, q75


def _filter_history(rows: list[dict[str, str]], *, label: str | None = None, method: str | None = None, bucket: str | None = None) -> list[dict[str, str]]:
    out = rows
    if label is not None:
        out = [row for row in out if row.get("label", "") == label]
    if method is not None:
        out = [row for row in out if row.get("method", "") == method]
    if bucket is not None:
        out = [row for row in out if row.get("bucket_id", "") == bucket]
    return out


def _best_summary_by_seed(rows: list[dict[str, str]], *, method: str | None = None) -> dict[tuple[str, str, str], dict[str, str]]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        if method is not None and row.get("method") != method:
            continue
        key = _seed_key(row)
        if key not in best or _f(row.get("best_score")) > _f(best[key].get("best_score")):
            best[key] = row
    return best


def _index_history(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str, str, str], list[dict[str, str]]]:
    indexed: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        indexed[_history_key(row)].append(row)
    for group in indexed.values():
        group.sort(key=lambda row: _x_value(row))
    return indexed


def _history_for_summaries(summary_rows: dict[tuple[str, str, str], dict[str, str]], history_index: dict[tuple[str, str, str, str, str, str], list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for summary in summary_rows.values():
        rows.extend(history_index.get(_summary_key(summary), []))
    return rows


def _plot_main_curves(cpu_hist: list[dict[str, str]], gpu_hist: list[dict[str, str]], gpu_rows: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    history_index = _index_history(gpu_hist)
    series = [
        ("Framsticks EA", cpu_hist, "#f59e0b"),
        ("Tree latent EA", _filter_history(gpu_hist, label="09_tree_vae_epoch80", method="latent_ea"), "#22d3ee"),
        ("Transformer latent EA", _filter_history(gpu_hist, label="10_transformer_vae_epoch90", method="latent_ea"), "#38bdf8"),
        ("Tree CMA-ES", _filter_history(gpu_hist, label="09_tree_vae_epoch80", method="cmaes"), "#a78bfa"),
        ("Best GPU per seed", _history_for_summaries(_best_summary_by_seed(gpu_rows), history_index), "#f472b6"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, rows, color in series:
        xs, med, q25, q75 = _aggregate_curve(rows)
        ax.plot(xs, med, color=color, linewidth=2.2, label=name)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.13, linewidth=0)
    ax.set_xlabel("Requested true evaluations per seed")
    ax.set_ylabel("Median best vertpos so far")
    ax.set_title("Evolution/optimization trajectories: median with interquartile band")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, output_dir, "06_history_median_curves.png", dpi)


def _plot_delta_curves(cpu_hist: list[dict[str, str]], gpu_hist: list[dict[str, str]], gpu_rows: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    history_index = _index_history(gpu_hist)
    series = [
        ("Framsticks EA", cpu_hist, "#f59e0b"),
        ("Tree latent EA", _filter_history(gpu_hist, label="09_tree_vae_epoch80", method="latent_ea"), "#22d3ee"),
        ("Transformer latent EA", _filter_history(gpu_hist, label="10_transformer_vae_epoch90", method="latent_ea"), "#38bdf8"),
        ("Best latent EA per seed", _history_for_summaries(_best_summary_by_seed([row for row in gpu_rows if row.get("method") == "latent_ea"]), history_index), "#34d399"),
        ("Best GPU per seed", _history_for_summaries(_best_summary_by_seed(gpu_rows), history_index), "#f472b6"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, rows, color in series:
        xs, med, q25, q75 = _aggregate_curve(rows, value_col="best_minus_seed_source")
        ax.plot(xs, med, color=color, linewidth=2.2, label=name)
        ax.fill_between(xs, q25, q75, color=color, alpha=0.13, linewidth=0)
    ax.axhline(0.0, color="#94a3b8", linewidth=1.0)
    ax.set_xlabel("Requested true evaluations per seed")
    ax.set_ylabel("Median improvement over start seed")
    ax.set_title("Improvement trajectories over starting genotype")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, output_dir, "07_history_delta_curves.png", dpi)


def _plot_bucket_facets(cpu_hist: list[dict[str, str]], gpu_hist: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    series = [
        ("Framsticks EA", lambda bucket: _filter_history(cpu_hist, method="frams", bucket=bucket), "#f59e0b"),
        ("Tree latent EA", lambda bucket: _filter_history(gpu_hist, label="09_tree_vae_epoch80", method="latent_ea", bucket=bucket), "#22d3ee"),
        ("Transformer latent EA", lambda bucket: _filter_history(gpu_hist, label="10_transformer_vae_epoch90", method="latent_ea", bucket=bucket), "#38bdf8"),
        ("Tree CMA-ES", lambda bucket: _filter_history(gpu_hist, label="09_tree_vae_epoch80", method="cmaes", bucket=bucket), "#a78bfa"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, bucket in zip(axes.ravel(), BUCKET_ORDER):
        for name, getter, color in series:
            xs, med, _, _ = _aggregate_curve(getter(bucket))
            ax.plot(xs, med, color=color, linewidth=1.8, label=name)
        ax.set_title(f"{bucket}: {BUCKET_LABELS[bucket]}")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Median best")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Median best-so-far by starting-fitness bucket", y=1.08)
    fig.tight_layout()
    return _save(fig, output_dir, "08_history_bucket_facets.png", dpi)


def _select_example_keys(cpu_rows: list[dict[str, str]], gpu_rows: list[dict[str, str]]) -> list[tuple[str, tuple[str, str, str]]]:
    baseline = {_seed_key(row): row for row in cpu_rows}
    best_gpu = _best_summary_by_seed(gpu_rows)
    diffs = []
    for key, row in best_gpu.items():
        base = baseline.get(key)
        if base is None:
            continue
        diffs.append((_f(row["best_score"]) - _f(base["best_score"]), key))
    if not diffs:
        return []
    diffs.sort()
    return [
        ("GPU loses most", diffs[0][1]),
        ("Typical", diffs[len(diffs) // 2][1]),
        ("GPU wins most", diffs[-1][1]),
    ]


def _plot_examples(cpu_rows: list[dict[str, str]], cpu_hist: list[dict[str, str]], gpu_rows: list[dict[str, str]], gpu_hist: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    cpu_index = _index_history(cpu_hist)
    gpu_index = _index_history(gpu_hist)
    baseline_summary = {_seed_key(row): row for row in cpu_rows}
    best_gpu_summary = _best_summary_by_seed(gpu_rows)
    tree_latent_summary = {key: row for key, row in _best_summary_by_seed([row for row in gpu_rows if row.get("label") == "09_tree_vae_epoch80" and row.get("method") == "latent_ea"]).items()}
    transformer_latent_summary = {key: row for key, row in _best_summary_by_seed([row for row in gpu_rows if row.get("label") == "10_transformer_vae_epoch90" and row.get("method") == "latent_ea"]).items()}
    examples = _select_example_keys(cpu_rows, gpu_rows)
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 4 * max(1, len(examples))), sharex=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, (title, key) in zip(axes, examples):
        configs = [
            ("Framsticks EA", baseline_summary.get(key), cpu_index, "#f59e0b"),
            ("Tree latent EA", tree_latent_summary.get(key), gpu_index, "#22d3ee"),
            ("Transformer latent EA", transformer_latent_summary.get(key), gpu_index, "#38bdf8"),
            ("Best GPU", best_gpu_summary.get(key), gpu_index, "#f472b6"),
        ]
        for name, summary, index, color in configs:
            if summary is None:
                continue
            rows = index.get(_summary_key(summary), [])
            if not rows:
                continue
            ax.plot([_x_value(row) for row in rows], [_f(row["best_score"]) for row in rows], color=color, linewidth=2.0, label=name)
        seed_score = _f(baseline_summary[key]["seed_source_fitness"])
        ax.axhline(seed_score, color="#94a3b8", linestyle="--", linewidth=1.0, label="start seed")
        ax.set_title(f"{title}: {key[0]} seed {key[1]} dataset {key[2]}")
        ax.set_ylabel("Best vertpos")
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Requested true evaluations per seed")
    fig.tight_layout()
    return _save(fig, output_dir, "09_history_example_seed_trajectories.png", dpi)


def _plot_winrate_over_time(cpu_rows: list[dict[str, str]], cpu_hist: list[dict[str, str]], gpu_rows: list[dict[str, str]], gpu_hist: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    cpu_index = _index_history(cpu_hist)
    gpu_index = _index_history(gpu_hist)
    baseline_summary = {_seed_key(row): row for row in cpu_rows}
    configs = [
        ("Tree latent EA", _best_summary_by_seed([row for row in gpu_rows if row.get("label") == "09_tree_vae_epoch80" and row.get("method") == "latent_ea"]), "#22d3ee"),
        ("Transformer latent EA", _best_summary_by_seed([row for row in gpu_rows if row.get("label") == "10_transformer_vae_epoch90" and row.get("method") == "latent_ea"]), "#38bdf8"),
        ("Best latent EA", _best_summary_by_seed([row for row in gpu_rows if row.get("method") == "latent_ea"]), "#34d399"),
        ("Best GPU", _best_summary_by_seed(gpu_rows), "#f472b6"),
    ]
    baseline_curves: dict[tuple[str, str, str], dict[int, float]] = {}
    for key, summary in baseline_summary.items():
        rows = cpu_index.get(_summary_key(summary), [])
        baseline_curves[key] = {_x_value(row): _f(row["best_score"]) for row in rows}
    xs = sorted({x for curve in baseline_curves.values() for x in curve})
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, summaries, color in configs:
        winrates = []
        for x in xs:
            wins = 0
            total = 0
            for key, summary in summaries.items():
                base_curve = baseline_curves.get(key)
                gpu_rows_for_key = gpu_index.get(_summary_key(summary), [])
                if not base_curve or not gpu_rows_for_key:
                    continue
                gpu_curve = {_x_value(row): _f(row["best_score"]) for row in gpu_rows_for_key}
                if x in base_curve and x in gpu_curve:
                    wins += gpu_curve[x] > base_curve[x]
                    total += 1
            winrates.append(wins / total if total else float("nan"))
        ax.plot(xs, winrates, color=color, linewidth=2.2, label=name)
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="#94a3b8", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Requested true evaluations per seed")
    ax.set_ylabel("Win rate vs Framsticks EA at same evaluation count")
    ax.set_title("Win rate over optimization budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, output_dir, "10_history_winrate_over_time.png", dpi)


def main() -> None:
    args = _build_parser().parse_args()
    cpu_run = _resolve(args.cpu_run)
    gpu_run = _resolve(args.gpu_run)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_rows = _load_rows(cpu_run, "cpu", history=False)
    gpu_rows = _load_rows(gpu_run, "gpu", history=False)
    cpu_hist = _load_rows(cpu_run, "cpu", history=True)
    gpu_hist = _load_rows(gpu_run, "gpu", history=True)
    if not cpu_rows or not gpu_rows or not cpu_hist or not gpu_hist:
        raise RuntimeError("Missing summary or history rows")

    paths = [
        _plot_main_curves(cpu_hist, gpu_hist, gpu_rows, output_dir, args.dpi),
        _plot_delta_curves(cpu_hist, gpu_hist, gpu_rows, output_dir, args.dpi),
        _plot_bucket_facets(cpu_hist, gpu_hist, output_dir, args.dpi),
        _plot_examples(cpu_rows, cpu_hist, gpu_rows, gpu_hist, output_dir, args.dpi),
        _plot_winrate_over_time(cpu_rows, cpu_hist, gpu_rows, gpu_hist, output_dir, args.dpi),
    ]
    summary_path = output_dir / "history_plot_inputs_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"cpu_run={cpu_run}",
                f"gpu_run={gpu_run}",
                f"cpu_summary_rows={len(cpu_rows)}",
                f"gpu_summary_rows={len(gpu_rows)}",
                f"cpu_history_rows={len(cpu_hist)}",
                f"gpu_history_rows={len(gpu_hist)}",
                "generated=",
                *[str(path) for path in paths],
            ]
        ),
        encoding="utf-8",
    )
    for path in paths:
        print(f"Wrote: {path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
