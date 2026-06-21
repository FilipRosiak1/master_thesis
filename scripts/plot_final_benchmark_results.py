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
    "": "Framsticks baseline",
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
    parser = argparse.ArgumentParser(description="Plot final Framsticks vs hidden-space benchmark results")
    parser.add_argument(
        "--cpu-run",
        default="exports/final_results_cpu/exports/final_seed_bucket_benchmark_baseline_20260613_011516",
    )
    parser.add_argument(
        "--gpu-run",
        default="exports/final_results_gpu/exports/final_seed_bucket_benchmark_20260607_194941",
    )
    parser.add_argument("--output-dir", default=f"exports/final_benchmark_plots_{stamp}")
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_job_rows(run_dir: Path, subdir: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((run_dir / "jobs" / subdir).glob("*.csv")):
        if path.name.endswith(".history.csv"):
            continue
        rows.extend(_read_csv(path))
    return rows


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _seed_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("bucket_id", ""), row.get("seed_rank", ""), row.get("seed_dataset_idx", ""))


def _label_name(label: str) -> str:
    return LABEL_NAMES.get(label, Path(label.replace("\\", "/")).name if "/" in label or "\\" in label else label)


def _method_name(method: str) -> str:
    return METHOD_NAMES.get(method, method)


def _mean(values: list[float]) -> float:
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else float("nan")


def _median(values: list[float]) -> float:
    values = sorted(value for value in values if not math.isnan(value))
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


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


def _config_rows(gpu_rows: list[dict[str, str]], label: str, method: str) -> list[dict[str, str]]:
    return [row for row in gpu_rows if row.get("label") == label and row.get("method") == method]


def _baseline_by_seed(cpu_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {_seed_key(row): row for row in cpu_rows}


def _gpu_best_by_seed(gpu_rows: list[dict[str, str]], *, method: str | None = None) -> dict[tuple[str, str, str], dict[str, str]]:
    best: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in gpu_rows:
        if method is not None and row.get("method") != method:
            continue
        key = _seed_key(row)
        if key not in best or _f(row.get("best_score")) > _f(best[key].get("best_score")):
            best[key] = row
    return best


def _boxplot_overview(cpu_rows: list[dict[str, str]], gpu_rows: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    configs = [
        ("Framsticks EA", cpu_rows),
        ("Tree latent EA", _config_rows(gpu_rows, "09_tree_vae_epoch80", "latent_ea")),
        ("Transformer latent EA", _config_rows(gpu_rows, "10_transformer_vae_epoch90", "latent_ea")),
        ("Tree CMA-ES", _config_rows(gpu_rows, "09_tree_vae_epoch80", "cmaes")),
        ("Fitness-aware CMA-ES", _config_rows(gpu_rows, FITNESS_AWARE_LABEL, "cmaes")),
        ("Best GPU per seed", list(_gpu_best_by_seed(gpu_rows).values())),
    ]
    data = [[_f(row["best_score"]) for row in rows] for _, rows in configs]
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops={"color": "#0f172a", "linewidth": 1.5})
    colors = ["#f59e0b", "#22d3ee", "#38bdf8", "#a78bfa", "#34d399", "#f472b6"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#e5e7eb")
    ax.set_xticklabels([name for name, _ in configs], rotation=25, ha="right")
    ax.set_ylabel("Best vertpos")
    ax.set_title("Final benchmark: distribution of best scores")
    ax.axhline(1.5, color="#64748b", linestyle="--", linewidth=1)
    fig.tight_layout()
    return _save(fig, output_dir, "01_best_score_boxplots.png", dpi)


def _winrate_heatmap(cpu_rows: list[dict[str, str]], gpu_rows: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    baseline = _baseline_by_seed(cpu_rows)
    labels = sorted({row["label"] for row in gpu_rows}, key=lambda x: list(LABEL_NAMES).index(x) if x in LABEL_NAMES else 99)
    methods = ["latent_ea", "cem", "cmaes"]
    winrates: list[list[float]] = []
    mean_diffs: list[list[float]] = []
    counts: list[list[int]] = []
    for label in labels:
        wr_row: list[float] = []
        diff_row: list[float] = []
        count_row: list[int] = []
        for method in methods:
            diffs = []
            for row in _config_rows(gpu_rows, label, method):
                base = baseline.get(_seed_key(row))
                if base is not None:
                    diffs.append(_f(row["best_score"]) - _f(base["best_score"]))
            wr_row.append(sum(diff > 0 for diff in diffs) / len(diffs) if diffs else float("nan"))
            diff_row.append(_mean(diffs))
            count_row.append(len(diffs))
        winrates.append(wr_row)
        mean_diffs.append(diff_row)
        counts.append(count_row)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    im0 = axes[0].imshow(winrates, vmin=0.0, vmax=1.0, cmap="viridis")
    im1 = axes[1].imshow(mean_diffs, vmin=-0.6, vmax=0.2, cmap="coolwarm")
    for ax, title, values, suffix in [
        (axes[0], "Win rate vs Framsticks EA", winrates, "%"),
        (axes[1], "Mean score difference vs Framsticks EA", mean_diffs, ""),
    ]:
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([_method_name(method) for method in methods], rotation=25, ha="right")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([_label_name(label) for label in labels])
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(labels)):
            for j in range(len(methods)):
                value = values[i][j]
                if math.isnan(value):
                    text = "n/a"
                elif suffix == "%":
                    text = f"{value * 100:.0f}%\nn={counts[i][j]}"
                else:
                    text = f"{value:+.3f}\nn={counts[i][j]}"
                ax.text(j, i, text, ha="center", va="center", color="#f8fafc", fontsize=8)
    fig.colorbar(im0, ax=axes[0], shrink=0.85)
    fig.colorbar(im1, ax=axes[1], shrink=0.85)
    fig.tight_layout()
    return _save(fig, output_dir, "02_winrate_and_mean_diff_heatmaps.png", dpi)


def _bucket_curves(cpu_rows: list[dict[str, str]], gpu_rows: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    series = [
        ("Framsticks EA", cpu_rows, "#f59e0b"),
        ("Tree latent EA", _config_rows(gpu_rows, "09_tree_vae_epoch80", "latent_ea"), "#22d3ee"),
        ("Transformer latent EA", _config_rows(gpu_rows, "10_transformer_vae_epoch90", "latent_ea"), "#38bdf8"),
        ("Tree CMA-ES", _config_rows(gpu_rows, "09_tree_vae_epoch80", "cmaes"), "#a78bfa"),
        ("Fitness-aware CMA-ES", _config_rows(gpu_rows, FITNESS_AWARE_LABEL, "cmaes"), "#34d399"),
        ("Best GPU per seed", list(_gpu_best_by_seed(gpu_rows).values()), "#f472b6"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(BUCKET_ORDER)))
    for name, rows, color in series:
        by_bucket: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_bucket[row["bucket_id"]].append(_f(row["best_score"]))
        means = [_mean(by_bucket[bucket]) for bucket in BUCKET_ORDER]
        ax.plot(x, means, marker="o", linewidth=2.2, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([BUCKET_LABELS[bucket] for bucket in BUCKET_ORDER])
    ax.set_xlabel("Starting seed fitness bucket")
    ax.set_ylabel("Mean best vertpos")
    ax.set_title("Mean best score by starting-fitness bucket")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save(fig, output_dir, "03_bucket_mean_scores.png", dpi)


def _paired_scatter(cpu_rows: list[dict[str, str]], gpu_rows: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    baseline = _baseline_by_seed(cpu_rows)
    configs = [
        ("Tree latent EA", _gpu_best_by_seed(_config_rows(gpu_rows, "09_tree_vae_epoch80", "latent_ea"))),
        ("Transformer latent EA", _gpu_best_by_seed(_config_rows(gpu_rows, "10_transformer_vae_epoch90", "latent_ea"))),
        ("Tree CMA-ES", _gpu_best_by_seed(_config_rows(gpu_rows, "09_tree_vae_epoch80", "cmaes"))),
        ("Best GPU per seed", _gpu_best_by_seed(gpu_rows)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (name, rows_by_seed) in zip(axes, configs):
        xs: list[float] = []
        ys: list[float] = []
        for key, row in rows_by_seed.items():
            base = baseline.get(key)
            if base is None:
                continue
            xs.append(_f(base["best_score"]))
            ys.append(_f(row["best_score"]))
        ax.scatter(xs, ys, s=18, alpha=0.75, color="#38bdf8", edgecolors="none")
        lo = min(xs + ys) if xs else 0.0
        hi = max(xs + ys) if xs else 1.0
        ax.plot([lo, hi], [lo, hi], color="#f59e0b", linestyle="--", linewidth=1.2)
        ax.set_title(name)
        ax.set_xlabel("Framsticks EA best")
        ax.set_ylabel("GPU method best")
        wins = sum(y > x for x, y in zip(xs, ys))
        ax.text(0.03, 0.92, f"wins {wins}/{len(xs)}", transform=ax.transAxes, fontsize=9)
    fig.suptitle("Paired per-seed comparison against Framsticks EA", y=0.995)
    fig.tight_layout()
    return _save(fig, output_dir, "04_paired_scatter_vs_baseline.png", dpi)


def _method_oracle_bars(cpu_rows: list[dict[str, str]], gpu_rows: list[dict[str, str]], output_dir: Path, dpi: int) -> Path:
    plt = _setup_matplotlib()
    baseline = _baseline_by_seed(cpu_rows)
    variants = [
        ("Best latent EA", _gpu_best_by_seed(gpu_rows, method="latent_ea")),
        ("Best CEM", _gpu_best_by_seed(gpu_rows, method="cem")),
        ("Best CMA-ES", _gpu_best_by_seed(gpu_rows, method="cmaes")),
        ("Best GPU overall", _gpu_best_by_seed(gpu_rows)),
    ]
    labels = []
    winrates = []
    mean_diffs = []
    med_diffs = []
    for name, rows_by_seed in variants:
        diffs = []
        for key, row in rows_by_seed.items():
            base = baseline.get(key)
            if base is not None:
                diffs.append(_f(row["best_score"]) - _f(base["best_score"]))
        labels.append(name)
        winrates.append(sum(diff > 0 for diff in diffs) / len(diffs))
        mean_diffs.append(_mean(diffs))
        med_diffs.append(_median(diffs))
    fig, ax1 = plt.subplots(figsize=(11, 5.8))
    x = list(range(len(labels)))
    bars = ax1.bar(x, winrates, color="#38bdf8", alpha=0.78, label="Win rate")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Win rate vs Framsticks EA")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    for bar, value in zip(bars, winrates):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value * 100:.0f}%", ha="center", va="bottom")
    ax2 = ax1.twinx()
    ax2.plot(x, mean_diffs, color="#f59e0b", marker="o", linewidth=2.4, label="Mean diff")
    ax2.plot(x, med_diffs, color="#f472b6", marker="s", linewidth=2.0, label="Median diff")
    ax2.axhline(0.0, color="#94a3b8", linewidth=1.0)
    ax2.set_ylabel("Score difference vs Framsticks EA")
    lines, line_labels = ax2.get_legend_handles_labels()
    ax1.legend([bars, *lines], ["Win rate", *line_labels], loc="upper left")
    ax1.set_title("Best GPU configuration per seed: oracle view")
    fig.tight_layout()
    return _save(fig, output_dir, "05_oracle_winrate_bars.png", dpi)


def main() -> None:
    args = _build_parser().parse_args()
    cpu_run = _resolve(args.cpu_run)
    gpu_run = _resolve(args.gpu_run)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cpu_rows = _load_job_rows(cpu_run, "cpu")
    gpu_rows = _load_job_rows(gpu_run, "gpu")
    if not cpu_rows:
        raise RuntimeError(f"No CPU rows found under {cpu_run}")
    if not gpu_rows:
        raise RuntimeError(f"No GPU rows found under {gpu_run}")

    paths = [
        _boxplot_overview(cpu_rows, gpu_rows, output_dir, args.dpi),
        _winrate_heatmap(cpu_rows, gpu_rows, output_dir, args.dpi),
        _bucket_curves(cpu_rows, gpu_rows, output_dir, args.dpi),
        _paired_scatter(cpu_rows, gpu_rows, output_dir, args.dpi),
        _method_oracle_bars(cpu_rows, gpu_rows, output_dir, args.dpi),
    ]
    summary_path = output_dir / "plot_inputs_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"cpu_run={cpu_run}",
                f"gpu_run={gpu_run}",
                f"cpu_rows={len(cpu_rows)}",
                f"gpu_rows={len(gpu_rows)}",
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
