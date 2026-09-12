#!/usr/bin/env python3
"""Generate the reproducible, non-CoSO LSO thesis figure suite.

The script deliberately reads raw benchmark jobs rather than merged analysis
tables.  Every plotted aggregate is also written as a round-trip-safe CSV.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "master_thesis" / "figures"
DATASET = ROOT / "datasets" / "f1" / "f1_dataset.txt"
RUN = (
    ROOT
    / "exports"
    / "final_results_gpu"
    / "exports"
    / "final_seed_bucket_benchmark_20260607_194941"
)
JOBS = RUN / "jobs" / "gpu"
MANIFEST = RUN / "seed_manifest.csv"
MINI = ROOT / "exports" / "final_results_cpu_mini" / "final_benchmark.csv"
MINI_HISTORY = ROOT / "exports" / "final_results_cpu_mini" / "final_benchmark.history.csv"
MINI_JOBS = ROOT / "exports" / "final_results_cpu_mini" / "jobs" / "cpu"
SELECTED = ROOT / "exports" / "f1_selected_10_ckpts_20260520_184738"
FITNESS_LOG = (
    ROOT
    / "exports"
    / "cluster_export_20260524_222538"
    / "models"
    / "f1_guided"
    / "fitness_aware_lhs_seed42"
    / "training.log"
)

EXPECTED_DATASET_N = 10_893
EXPECTED_DATASET_MAX = 2.2952309520191614
PHENOTYPE_GENOTYPE = "LX(FfXrLfXq((X,LXLCXXfFX)), mXLqQFX)"
BUCKETS = ("B1", "B2", "B3", "B4", "B5", "B6")
BUCKET_LIMITS = {
    "B1": (0.3, 0.5),
    "B2": (0.5, 0.75),
    "B3": (0.75, 1.0),
    "B4": (1.0, 1.2),
    "B5": (1.2, 1.5),
    "B6": (1.5, EXPECTED_DATASET_MAX),
}
MODELS = (
    "09_tree_vae_epoch80",
    "10_transformer_vae_epoch90",
    "02_lhs_latent256_seed42_best",
    "06_lhs_depth_seed42_best",
    "fitness_aware_lhs_seed42",
)
MODEL_NAMES = {
    "09_tree_vae_epoch80": "TreeVAE-80",
    "10_transformer_vae_epoch90": "TransformerVAE-90",
    "02_lhs_latent256_seed42_best": "LHS-256",
    "06_lhs_depth_seed42_best": "LHS-depth",
    "fitness_aware_lhs_seed42": "Fitness-aware LHS",
}
METHODS = ("latent_ea", "cem", "cmaes")
METHOD_NAMES = {"latent_ea": "Latent EA", "cem": "CEM", "cmaes": "CMA-ES"}
SELECTED_EPOCHS = {
    "09_tree_vae_epoch80": 80,
    "10_transformer_vae_epoch90": 90,
    "02_lhs_latent256_seed42_best": 720,
    "06_lhs_depth_seed42_best": 1170,
    "fitness_aware_lhs_seed42": 440,
}
TRAINING_LOGS = {
    model: (
        FITNESS_LOG
        if model == "fitness_aware_lhs_seed42"
        else SELECTED / model / "training.log"
    )
    for model in MODELS
}
SERIES = (
    ("tree_lso", "TreeVAE-80 + latent EA"),
    ("transformer_lso", "TransformerVAE-90 + latent EA"),
    ("mini_ea", "Mini Framsticks EA"),
)
SERIES_COLORS = {"tree_lso": "#0072B2", "transformer_lso": "#D55E00", "mini_ea": "#009E73"}
LSO_SIMULATION_CHAIN = ("eval-allcriteria.sim",)
MINI_SIMULATION_CHAIN = (
    "eval-allcriteria-mini.sim",
    "deterministic.sim",
    "sample-period-2.sim",
    "only-body.sim",
)
LSO_SIMULATION_CHAIN_TEXT = ";".join(LSO_SIMULATION_CHAIN)
MINI_SIMULATION_CHAIN_TEXT = ";".join(MINI_SIMULATION_CHAIN)
COMPARISON_SCOPE = "cross-configuration descriptive contrast; not a controlled method effect"
LSO_SOURCE_ARTIFACTS = "jobs/gpu/*.csv; jobs/gpu/*.history.csv; jobs/gpu/*.details.json"
MINI_SOURCE_ARTIFACTS = "final_benchmark.csv; final_benchmark.history.csv; jobs/cpu/*.details.json"
MODEL_COLORS = {
    "09_tree_vae_epoch80": "#0072B2",
    "10_transformer_vae_epoch90": "#D55E00",
    "02_lhs_latent256_seed42_best": "#009E73",
    "06_lhs_depth_seed42_best": "#CC79A7",
    "fitness_aware_lhs_seed42": "#E69F00",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.edgecolor": "#4D4D4D",
            "axes.linewidth": 0.7,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.linestyle": ":",
            "lines.linewidth": 1.35,
            "legend.frameon": False,
            "ps.fonttype": 42,
        }
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


DETAIL_SIM_RE = re.compile(r'^\s*"framsticks_sim":\s*"([^"]+)"')


def details_simulation_chain(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            match = DETAIL_SIM_RE.match(line)
            if match:
                chain = tuple(Path(item).name for item in match.group(1).split(";"))
                require(all(chain), f"Empty simulation-chain entry in {path}")
                return chain
            if line_number >= 100:
                break
    raise ValueError(f"Missing framsticks_sim in the header of {path}")


def validate_simulation_provenance() -> None:
    lso_details = sorted(JOBS.glob("*.details.json"))
    mini_details = sorted(MINI_JOBS.glob("*.details.json"))
    require(len(lso_details) == 90, f"Found {len(lso_details)} LSO details files, expected 90")
    require(len(mini_details) == 6, f"Found {len(mini_details)} mini details files, expected 6")
    lso_stems = {path.name.removesuffix(".csv") for path in JOBS.glob("*.csv") if not path.name.endswith(".history.csv")}
    lso_history_stems = {path.name.removesuffix(".history.csv") for path in JOBS.glob("*.history.csv")}
    lso_details_stems = {path.name.removesuffix(".details.json") for path in lso_details}
    mini_stems = {path.name.removesuffix(".csv") for path in MINI_JOBS.glob("*.csv") if not path.name.endswith(".history.csv")}
    mini_history_stems = {path.name.removesuffix(".history.csv") for path in MINI_JOBS.glob("*.history.csv")}
    mini_details_stems = {path.name.removesuffix(".details.json") for path in mini_details}
    require(lso_stems == lso_history_stems == lso_details_stems, "LSO result, history, and details stems differ")
    require(mini_stems == mini_history_stems == mini_details_stems, "Mini result, history, and details stems differ")
    for path in lso_details:
        require(
            details_simulation_chain(path) == LSO_SIMULATION_CHAIN,
            f"Unexpected LSO simulation chain in {path}",
        )
    for path in mini_details:
        require(
            details_simulation_chain(path) == MINI_SIMULATION_CHAIN,
            f"Unexpected mini simulation chain in {path}",
        )


def finite_float(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    require(math.isfinite(result), f"Non-finite {field}: {value!r}")
    return result


def close(a: float, b: float, tolerance: float = 1e-7) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tolerance)


def fcsv(value: float) -> str:
    return format(float(value), ".17g")


def write_csv(name: str, fields: list[str], rows: list[dict[str, object]]) -> Path:
    path = FIGURES / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return path


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, format="eps", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return path


def lighten(hex_color: str, amount: float = 0.76) -> tuple[float, float, float]:
    rgb = np.asarray(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * amount)


def bucket_label(bucket: str, multiline: bool = False) -> str:
    lo, hi = BUCKET_LIMITS[bucket]
    ranges = {
        "B1": "0.300-0.500",
        "B2": "0.500-0.750",
        "B3": "0.750-1.000",
        "B4": "1.000-1.200",
        "B5": "1.200-1.500",
        "B6": "1.500-2.295",
    }
    separator = "\n" if multiline else " "
    return f"{bucket}{separator}{ranges[bucket]}"


def key_of(row: dict[str, str]) -> tuple[str, int, int]:
    return row["bucket_id"], int(row["seed_rank"]), int(row["seed_dataset_idx"])


def normalized_genotype(genotype: str) -> str:
    return "".join(genotype.split())


def canonical_model_label(label: str) -> str:
    if label == "exports/cluster_export_20260524_222538/models/f1_guided/fitness_aware_lhs_seed42":
        return "fitness_aware_lhs_seed42"
    return label


def retained(row: dict[str, str]) -> tuple[float, float, bool, str]:
    best = finite_float(row["best_score"], "best_score")
    source = finite_float(row["seed_source_fitness"], "seed_source_fitness")
    score = max(best, source)
    stored = finite_float(row["best_or_source_score"], "best_or_source_score")
    require(close(score, stored), "Stored retained score disagrees with max(best_score, source)")
    improvement = score - source
    stored_imp = finite_float(
        row["best_or_source_minus_seed_source"], "best_or_source_minus_seed_source"
    )
    require(close(improvement, stored_imp), "Stored retained improvement disagrees with arithmetic")
    source_retained = source >= best
    genotype = row["seed_genotype"] if source_retained else row["best_genotype"]
    require(bool(genotype), "Retained genotype is empty")
    return score, improvement, source_retained, genotype


def sample_sd(values: list[float]) -> float:
    require(len(values) >= 2, "Sample SD requires at least two values")
    result = statistics.stdev(values)
    require(math.isfinite(result), "Non-finite sample SD")
    return result


def quantiles(values: list[float]) -> tuple[float, float, float]:
    q1, median, q3 = np.quantile(np.asarray(values), [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def load_manifest(dataset: list[tuple[str, str, float]]) -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 150, f"Manifest has {len(rows)} rows, expected 150")
    seen: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (row["bucket_id"], int(row["seed_rank"]), int(row["dataset_idx"]))
        require(key not in seen, f"Duplicate manifest key {key}")
        seen.add(key)
        require(row["bucket_id"] in BUCKETS, f"Unknown bucket {row['bucket_id']}")
        lo, hi = BUCKET_LIMITS[row["bucket_id"]]
        expected_max = "" if row["bucket_id"] == "B6" else fcsv(hi)
        require(close(finite_float(row["bucket_min"], "bucket_min"), lo), "Manifest bucket min mismatch")
        if row["bucket_id"] == "B6":
            require(row["bucket_max"] == "", "B6 manifest maximum must be open in source data")
        else:
            require(close(finite_float(row["bucket_max"], "bucket_max"), hi), "Manifest bucket max mismatch")
        idx = int(row["dataset_idx"])
        require(0 <= idx < len(dataset), f"Manifest dataset index out of range: {idx}")
        genotype, _, fitness = dataset[idx]
        require(
            row["genotype"] == normalized_genotype(genotype),
            f"Whitespace-normalized manifest genotype mismatch at dataset index {idx}",
        )
        require(close(finite_float(row["fitness"], "manifest fitness"), fitness), f"Manifest fitness mismatch at {idx}")
    counts = {bucket: sum(row["bucket_id"] == bucket for row in rows) for bucket in BUCKETS}
    require(all(count == 25 for count in counts.values()), f"Manifest bucket counts are {counts}")
    return rows


def generate_pipeline() -> list[Path]:
    stages = (
        "Source genotype",
        "Autoencoder encoder",
        "Initial latent vector",
        "Latent optimizer",
        "Decoder",
        "Syntax/duplicate checks",
        "Framsticks evaluation",
        "Optimizer update",
    )
    fig, ax = plt.subplots(figsize=(4.5, 7.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ys = np.linspace(0.92, 0.08, len(stages))
    colors = ("#E8F1F8", "#E8F1F8", "#F4F4F4", "#FDEBD0", "#E8F1F8", "#F4F4F4", "#E5F3EC", "#FDEBD0")
    for index, (stage, y, color) in enumerate(zip(stages, ys, colors)):
        box = FancyBboxPatch(
            (0.22, y - 0.038),
            0.56,
            0.076,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            facecolor=color,
            edgecolor="#3F4A54",
            linewidth=0.85,
        )
        ax.add_patch(box)
        ax.text(0.5, y, stage, ha="center", va="center", fontsize=9, fontfamily="sans-serif")
        if index:
            ax.add_patch(
                FancyArrowPatch(
                    (0.5, ys[index - 1] - 0.040),
                    (0.5, y + 0.040),
                    arrowstyle="-|>",
                    mutation_scale=9,
                    color="#4D4D4D",
                    linewidth=0.9,
                )
            )
    feedback_x = 0.88
    ax.plot(
        [0.79, feedback_x, feedback_x],
        [ys[-1], ys[-1], ys[3]],
        color="#B34D2E",
        linewidth=1.1,
    )
    ax.add_patch(
        FancyArrowPatch(
            (feedback_x, ys[3]),
            (0.79, ys[3]),
            arrowstyle="-|>",
            mutation_scale=10,
            color="#B34D2E",
            linewidth=1.1,
        )
    )
    ax.text(0.93, (ys[-1] + ys[3]) / 2, "feedback", rotation=90, ha="center", va="center", color="#8F3D27", fontfamily="sans-serif")
    return [save_figure(fig, "lso_pipeline.eps")]


def load_dataset() -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    with DATASET.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.rstrip("\n\r").rsplit("\t", 1)
            require(len(parts) == 2 and parts[0], f"Malformed dataset row {line_number}")
            rows.append((parts[0], parts[1], finite_float(parts[1], f"dataset fitness row {line_number}")))
    require(len(rows) == EXPECTED_DATASET_N, f"Dataset has {len(rows)} rows, expected {EXPECTED_DATASET_N}")
    actual_max = max(row[2] for row in rows)
    require(actual_max == EXPECTED_DATASET_MAX, f"Dataset maximum is {actual_max!r}, expected {EXPECTED_DATASET_MAX!r}")
    return rows


def generate_dataset_figure(
    dataset: list[tuple[str, str, float]], manifest: list[dict[str, str]]
) -> list[Path]:
    manifest_by_idx = {int(row["dataset_idx"]): row for row in manifest}
    csv_rows = []
    for idx, (genotype, source_fitness, fitness) in enumerate(dataset):
        selected = manifest_by_idx.get(idx)
        csv_rows.append(
            {
                "dataset_idx": idx,
                "genotype": genotype,
                "fitness": source_fitness,
                "selected_manifest_source": "yes" if selected else "no",
                "bucket_id": selected["bucket_id"] if selected else "",
                "seed_rank": selected["seed_rank"] if selected else "",
                "dataset_count_metadata": EXPECTED_DATASET_N,
                "exact_dataset_max_metadata": fcsv(EXPECTED_DATASET_MAX),
            }
        )
    csv_path = write_csv(
        "dataset_fitness_buckets.csv",
        ["dataset_idx", "genotype", "fitness", "selected_manifest_source", "bucket_id", "seed_rank", "dataset_count_metadata", "exact_dataset_max_metadata"],
        csv_rows,
    )
    values = np.asarray([row[2] for row in dataset])
    bins = np.linspace(values.min(), EXPECTED_DATASET_MAX, 61)
    counts, edges = np.histogram(values, bins=bins)
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    bucket_colors = ["#BDBDBD", "#56B4E9", "#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        center = (left + right) / 2
        index = 0 if center < 0.3 else min(6, 1 + sum(center >= BUCKET_LIMITS[b][1] for b in BUCKETS[:-1]))
        ax.bar(center, count, width=right - left, color=bucket_colors[index], edgecolor="white", linewidth=0.25)
    for boundary in (0.3, 0.5, 0.75, 1.0, 1.2, 1.5, EXPECTED_DATASET_MAX):
        ax.axvline(boundary, color="#555555", linewidth=0.7, linestyle="--")
    for bucket in BUCKETS:
        lo, hi = BUCKET_LIMITS[bucket]
        ax.text((lo + hi) / 2, max(counts) * 1.025, bucket, ha="center", va="bottom", fontsize=7.5)
    ax.text(0.15, max(counts) * 1.025, "below B1", ha="center", va="bottom", fontsize=7.5, color="#555555")
    ax.set_xlim(values.min(), EXPECTED_DATASET_MAX + 0.015)
    ax.set_ylim(0, max(counts) * 1.14)
    ax.set_xlabel("Dataset fitness")
    ax.set_ylabel("Genotypes")
    ax.set_title("Dataset fitness and benchmark source buckets")
    ax.grid(axis="x", visible=False)
    ax.text(
        0.985,
        0.875,
        f"n = {len(values):,}; max = {EXPECTED_DATASET_MAX:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    eps_path = save_figure(fig, "dataset_fitness_buckets.eps")
    return [eps_path, csv_path]


VAL_RE = re.compile(
    r"^ValRecon\s+(\d+)/(\d+)\s+Exact:\s*(\d+)/(\d+)\s+ExactAcc:\s*([0-9.]+)%\s+AvgSim:\s*([0-9.]+)%"
)


def generate_training_curves() -> list[Path]:
    parsed: dict[str, dict[int, tuple[int, int, int, float, float]]] = {}
    for model, path in TRAINING_LOGS.items():
        require(path.is_file(), f"Missing selected training log: {path}")
        epochs: dict[int, tuple[int, int, int, float, float]] = {}
        with path.open(encoding="utf-8", errors="strict") as handle:
            for line in handle:
                match = VAL_RE.match(line)
                if not match:
                    continue
                epoch, total, exact, n, exact_pct, similarity = match.groups()
                record = (int(total), int(exact), int(n), float(exact_pct), float(similarity))
                epochs[int(epoch)] = record
        require(epochs, f"No ValRecon rows in {path}")
        require(SELECTED_EPOCHS[model] in epochs, f"Selected epoch missing from {path}")
        require(all(record[2] == 2178 for record in epochs.values()), f"Unexpected validation denominator in {path}")
        for epoch, (_, exact, n, exact_pct, _) in epochs.items():
            require(abs(100.0 * exact / n - exact_pct) <= 0.006, f"Exact percentage mismatch at {model} epoch {epoch}")
        parsed[model] = epochs
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for epoch, (total, exact, n, exact_pct, similarity) in sorted(parsed[model].items()):
            rows.append(
                {
                    "model": model,
                    "model_name": MODEL_NAMES[model],
                    "epoch": epoch,
                    "training_epochs_planned": total,
                    "validation_exact_count": exact,
                    "validation_n": n,
                    "validation_exact_percent": fcsv(exact_pct),
                    "average_string_similarity_percent": fcsv(similarity),
                    "selected_epoch": "yes" if epoch == SELECTED_EPOCHS[model] else "no",
                    "source_log": str(TRAINING_LOGS[model].relative_to(ROOT)).replace("\\", "/"),
                }
            )
    csv_path = write_csv(
        "autoencoder_training_curves.csv",
        ["model", "model_name", "epoch", "training_epochs_planned", "validation_exact_count", "validation_n", "validation_exact_percent", "average_string_similarity_percent", "selected_epoch", "source_log"],
        rows,
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55), sharex=True)
    for model in MODELS:
        epochs = sorted(parsed[model])
        exact = [parsed[model][epoch][3] for epoch in epochs]
        similarity = [parsed[model][epoch][4] for epoch in epochs]
        color = MODEL_COLORS[model]
        axes[0].plot(epochs, exact, color=color, label=MODEL_NAMES[model])
        axes[1].plot(epochs, similarity, color=color, label=MODEL_NAMES[model])
        selected = SELECTED_EPOCHS[model]
        axes[0].scatter([selected], [parsed[model][selected][3]], color=color, edgecolor="white", linewidth=0.5, s=28, zorder=4)
        axes[1].scatter([selected], [parsed[model][selected][4]], color=color, edgecolor="white", linewidth=0.5, s=28, zorder=4)
    axes[0].set_title("Exact reconstruction")
    axes[1].set_title("Average string similarity")
    axes[0].set_ylabel("Validation score (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_xlim(0, 1200)
        ax.set_ylim(bottom=0)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Autoencoder validation curves", y=1.01)
    fig.tight_layout(rect=(0, 0.14, 1, 0.98))
    return [save_figure(fig, "autoencoder_training_curves.eps"), csv_path]


def load_endpoints(
    manifest: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[tuple[str, int, int, str, str], dict[str, str]]]:
    files = sorted(JOBS.glob("*.csv"))
    files = [path for path in files if not path.name.endswith(".history.csv")]
    require(len(files) == 90, f"Found {len(files)} raw final LSO CSVs, expected 90")
    rows: list[dict[str, str]] = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            part = list(csv.DictReader(handle))
        require(len(part) == 25, f"{path.name} has {len(part)} rows, expected 25")
        rows.extend(part)
    require(len(rows) == 2250, f"LSO endpoints have {len(rows)} rows, expected 2250")
    manifest_keys = {(row["bucket_id"], int(row["seed_rank"]), int(row["dataset_idx"])): row for row in manifest}
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_full_key: dict[tuple[str, int, int, str, str], dict[str, str]] = {}
    for row in rows:
        row["label"] = canonical_model_label(row["label"])
        require(row["label"] in MODELS and row["method"] in METHODS and row["bucket_id"] in BUCKETS, "Unexpected endpoint configuration")
        seed_key = key_of(row)
        require(seed_key in manifest_keys, f"LSO endpoint key is absent from manifest: {seed_key}")
        source = manifest_keys[seed_key]
        require(row["seed_genotype"] == source["genotype"], f"LSO source genotype mismatch for {seed_key}")
        require(close(finite_float(row["seed_source_fitness"], "LSO source fitness"), finite_float(source["fitness"], "manifest fitness")), f"LSO source fitness mismatch for {seed_key}")
        retained(row)
        groups[(row["label"], row["method"])].append(row)
        full_key = (*key_of(row), row["label"], row["method"])
        require(full_key not in by_full_key, f"Duplicate endpoint key {full_key}")
        by_full_key[full_key] = row
    require(len(groups) == 15, f"Found {len(groups)} endpoint groups, expected 15")
    for group, part in groups.items():
        require(len(part) == 150, f"Endpoint group {group} has {len(part)} rows")
        require({key_of(row) for row in part} == set(manifest_keys), f"Endpoint group {group} does not exactly match the manifest")
        bucket_counts = {bucket: sum(row["bucket_id"] == bucket for row in part) for bucket in BUCKETS}
        require(all(n == 25 for n in bucket_counts.values()), f"Endpoint group {group} bucket counts {bucket_counts}")
    return rows, by_full_key


def endpoint_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    result: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[(row["label"], row["method"])].append(row)
    return result


def generate_matrix_and_distributions(rows: list[dict[str, str]]) -> list[Path]:
    groups = endpoint_groups(rows)
    matrix_rows: list[dict[str, object]] = []
    score_matrix = np.zeros((len(MODELS), len(METHODS)))
    improvement_matrix = np.zeros_like(score_matrix)
    score_sd = np.zeros_like(score_matrix)
    improvement_sd = np.zeros_like(score_matrix)
    distribution_rows: list[dict[str, object]] = []
    for i, model in enumerate(MODELS):
        for j, method in enumerate(METHODS):
            scores = [retained(row)[0] for row in groups[(model, method)]]
            improvements = [retained(row)[1] for row in groups[(model, method)]]
            score_matrix[i, j] = statistics.mean(scores)
            improvement_matrix[i, j] = statistics.mean(improvements)
            score_sd[i, j] = sample_sd(scores)
            improvement_sd[i, j] = sample_sd(improvements)
            matrix_rows.append(
                {
                    "model": model,
                    "model_name": MODEL_NAMES[model],
                    "method": method,
                    "method_name": METHOD_NAMES[method],
                    "n": 150,
                    "mean_retained_score": fcsv(score_matrix[i, j]),
                    "sample_sd_retained_score": fcsv(score_sd[i, j]),
                    "mean_retained_improvement": fcsv(improvement_matrix[i, j]),
                    "sample_sd_retained_improvement": fcsv(improvement_sd[i, j]),
                    "retained_policy": "max(best_score,seed_source_fitness)",
                    "simulation_chain": LSO_SIMULATION_CHAIN_TEXT,
                    "source_artifacts": LSO_SOURCE_ARTIFACTS,
                }
            )
            q1, median, q3 = quantiles(scores)
            distribution_rows.append(
                {
                    "model": model,
                    "model_name": MODEL_NAMES[model],
                    "method": method,
                    "method_name": METHOD_NAMES[method],
                    "n": 150,
                    "mean": fcsv(statistics.mean(scores)),
                    "sample_sd": fcsv(sample_sd(scores)),
                    "median": fcsv(median),
                    "q1": fcsv(q1),
                    "q3": fcsv(q3),
                    "max": fcsv(max(scores)),
                    "metric": "retained_score=max(best_score,seed_source_fitness)",
                    "simulation_chain": LSO_SIMULATION_CHAIN_TEXT,
                    "source_artifacts": LSO_SOURCE_ARTIFACTS,
                }
            )
    matrix_csv = write_csv(
        "lso_checkpoint_optimizer_matrix.csv",
        ["model", "model_name", "method", "method_name", "n", "mean_retained_score", "sample_sd_retained_score", "mean_retained_improvement", "sample_sd_retained_improvement", "retained_policy", "simulation_chain", "source_artifacts"],
        matrix_rows,
    )
    score_cmap = LinearSegmentedColormap.from_list("score", ["#F7FBFF", "#6BAED6", "#08519C"])
    imp_cmap = LinearSegmentedColormap.from_list("improvement", ["#FFF7EC", "#FDAE6B", "#A63603"])
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0))
    for ax, data, sd, title, cmap in (
        (axes[0], score_matrix, score_sd, "Mean retained score", score_cmap),
        (axes[1], improvement_matrix, improvement_sd, "Mean retained improvement", imp_cmap),
    ):
        image = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.grid(False)
        ax.set_xticks(range(len(METHODS)), [METHOD_NAMES[m] for m in METHODS])
        ax.set_yticks(range(len(MODELS)), [MODEL_NAMES[m] for m in MODELS])
        ax.set_title(title)
        threshold = (data.min() + data.max()) / 2
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}\n$\\pm${sd[i, j]:.3f}", ha="center", va="center", fontsize=7.1, color="white" if data[i, j] > threshold else "#222222")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    matrix_eps = save_figure(fig, "lso_checkpoint_optimizer_matrix.eps")

    distribution_csv = write_csv(
        "lso_final_score_distributions.csv",
        ["model", "model_name", "method", "method_name", "n", "mean", "sample_sd", "median", "q1", "q3", "max", "metric", "simulation_chain", "source_artifacts"],
        distribution_rows,
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.5), sharey=True)
    for ax, method in zip(axes, METHODS):
        values = [[retained(row)[0] for row in groups[(model, method)]] for model in MODELS]
        plot = ax.boxplot(values, patch_artist=True, widths=0.62, showfliers=True, flierprops={"marker": ".", "markersize": 2.5, "markerfacecolor": "#555555", "markeredgecolor": "#555555"})
        for patch, model in zip(plot["boxes"], MODELS):
            patch.set_facecolor(lighten(MODEL_COLORS[model], 0.64))
            patch.set_edgecolor(MODEL_COLORS[model])
        for median_line in plot["medians"]:
            median_line.set_color("#222222")
            median_line.set_linewidth(1.2)
        ax.set_xticks(range(1, 6), [MODEL_NAMES[m] for m in MODELS], rotation=35, ha="right")
        ax.set_title(METHOD_NAMES[method])
        ax.set_xlabel("Checkpoint")
    axes[0].set_ylabel("Retained final score")
    fig.suptitle("LSO retained-score distributions", y=1.01)
    fig.tight_layout()
    distribution_eps = save_figure(fig, "lso_final_score_distributions.eps")
    return [matrix_eps, matrix_csv, distribution_eps, distribution_csv]


def history_summary(
    paths: list[Path],
    endpoints: dict[tuple[str, int, int, str, str], dict[str, str]],
) -> tuple[dict[tuple[str, str, int], list[float]], int]:
    values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    total_rows = 0
    for path in paths:
        file_configurations: set[tuple[str, str, str]] = set()
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            per_seed: dict[tuple[str, int, int], tuple[int, float, int]] = {}
            file_rows = 0
            for row in reader:
                file_rows += 1
                total_rows += 1
                model, method = canonical_model_label(row["label"]), row["method"]
                require(model in MODELS and method in METHODS, f"Unexpected history configuration in {path.name}")
                file_configurations.add((model, method, row["bucket_id"]))
                seed_key = key_of(row)
                step = int(row["step"])
                slots = int(row["evaluations_requested"])
                require(slots == step * 100, f"Requested slots mismatch in {path.name}")
                source = finite_float(row["seed_source_fitness"], "history source fitness")
                best = finite_float(row["best_score"], "history best score")
                score = max(best, source)
                require(close(score, finite_float(row["best_or_source_score"], "history retained score")), f"History retained arithmetic mismatch in {path.name}")
                previous = per_seed.get(seed_key)
                if previous is None:
                    require(step == 1, f"History for {seed_key} does not start at step 1")
                else:
                    require(step == previous[0] + 1, f"Non-contiguous history for {seed_key}")
                    require(score + 1e-10 >= previous[1], f"Retained best-so-far decreases for {seed_key}")
                per_seed[seed_key] = (step, score, slots)
                values[(model, method, slots)].append(score)
        require(file_rows == 7500, f"{path.name} has {file_rows} rows, expected 7500")
        require(len(file_configurations) == 1, f"{path.name} mixes configurations")
        require(len(per_seed) == 25, f"{path.name} has {len(per_seed)} sources, expected 25")
        for seed_key, (step, score, slots) in per_seed.items():
            require(step == 300 and slots == 30_000, f"History endpoint missing for {seed_key}")
            endpoint = endpoints[(*seed_key, model, method)]
            require(close(score, retained(endpoint)[0]), f"History endpoint differs from final row for {seed_key}, {model}, {method}")
    return values, total_rows


def generate_convergence(
    endpoint_lookup: dict[tuple[str, int, int, str, str], dict[str, str]]
) -> tuple[list[Path], dict[tuple[str, str, int], list[float]]]:
    paths = sorted(JOBS.glob("*.history.csv"))
    require(len(paths) == 90, f"Found {len(paths)} raw LSO histories, expected 90")
    values, total = history_summary(paths, endpoint_lookup)
    require(total == 675_000, f"LSO histories have {total} rows, expected 675000")
    require(len(values) == 15 * 300, f"LSO history has {len(values)} aggregate slots")
    rows: list[dict[str, object]] = []
    for model in MODELS:
        for method in METHODS:
            for slots in range(100, 30_001, 100):
                scores = values[(model, method, slots)]
                require(len(scores) == 150, f"History group {(model, method, slots)} has {len(scores)} values")
                q1, median, q3 = quantiles(scores)
                rows.append({"model": model, "model_name": MODEL_NAMES[model], "method": method, "method_name": METHOD_NAMES[method], "requested_candidate_slots": slots, "n": 150, "q1_retained_score": fcsv(q1), "median_retained_score": fcsv(median), "q3_retained_score": fcsv(q3), "metric": "retained_best_so_far=max(best_score,seed_source_fitness)", "simulation_chain": LSO_SIMULATION_CHAIN_TEXT, "source_artifacts": LSO_SOURCE_ARTIFACTS})
    csv_path = write_csv("lso_convergence.csv", ["model", "model_name", "method", "method_name", "requested_candidate_slots", "n", "q1_retained_score", "median_retained_score", "q3_retained_score", "metric", "simulation_chain", "source_artifacts"], rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.25), sharex=True, sharey=True)
    x = np.arange(100, 30_001, 100)
    for ax, method in zip(axes, METHODS):
        for model in MODELS:
            summaries = [quantiles(values[(model, method, int(slot))]) for slot in x]
            median = np.asarray([item[1] for item in summaries])
            color = MODEL_COLORS[model]
            ax.plot(x, median, color=color, label=MODEL_NAMES[model], zorder=2)
        ax.set_title(METHOD_NAMES[method])
        ax.set_xlabel("Requested candidate slots")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    axes[0].set_ylabel("Retained best-so-far score")
    axes[-1].legend(loc="lower right")
    fig.suptitle("LSO convergence", y=1.01)
    fig.tight_layout()
    return [save_figure(fig, "lso_convergence.eps"), csv_path], values


def generate_bucket_improvement(rows: list[dict[str, str]]) -> list[Path]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["label"], row["method"], row["bucket_id"])].append(row)
    data = np.zeros((15, 6))
    csv_rows: list[dict[str, object]] = []
    row_labels: list[str] = []
    row_index = 0
    for model in MODELS:
        for method in METHODS:
            row_labels.append(f"{MODEL_NAMES[model]} / {METHOD_NAMES[method]}")
            for col, bucket in enumerate(BUCKETS):
                part = grouped[(model, method, bucket)]
                require(len(part) == 25, f"Bucket cell {(model, method, bucket)} has {len(part)} rows")
                scores = [retained(row)[0] for row in part]
                improvements = [retained(row)[1] for row in part]
                source_count = sum(retained(row)[2] for row in part)
                embedded_count = sum(
                    method == "latent_ea" and row.get("best_is_source_seed", "").lower() == "true"
                    for row in part
                )
                posthoc_count = sum(
                    method != "latent_ea"
                    and finite_float(row["seed_source_fitness"], "source")
                    > finite_float(row["best_score"], "best")
                    for row in part
                )
                tie_count = sum(
                    finite_float(row["seed_source_fitness"], "source")
                    == finite_float(row["best_score"], "best")
                    for row in part
                )
                if method == "latent_ea":
                    require(source_count == embedded_count == tie_count, f"Embedded-source metadata mismatch for {(model, method, bucket)}")
                else:
                    require(source_count == posthoc_count + tie_count, f"Post-hoc source metadata mismatch for {(model, method, bucket)}")
                data[row_index, col] = statistics.mean(improvements)
                csv_rows.append(
                    {
                        "metric_policy": "retained_source_max",
                        "retained_improvement_definition": "max(best_score,seed_source_fitness)-seed_source_fitness",
                        "model": model,
                        "model_name": MODEL_NAMES[model],
                        "method": method,
                        "method_name": METHOD_NAMES[method],
                        "bucket": bucket,
                        "bucket_min": fcsv(BUCKET_LIMITS[bucket][0]),
                        "bucket_max_finite": fcsv(BUCKET_LIMITS[bucket][1]),
                        "dataset_exact_max_metadata": fcsv(EXPECTED_DATASET_MAX),
                        "n": 25,
                        "mean_retained_score": fcsv(statistics.mean(scores)),
                        "sample_sd_retained_score": fcsv(sample_sd(scores)),
                        "max_retained_score": fcsv(max(scores)),
                        "mean_retained_improvement": fcsv(statistics.mean(improvements)),
                        "sample_sd_retained_improvement": fcsv(sample_sd(improvements)),
                        "source_retained_count": source_count,
                        "embedded_stored_source_count": embedded_count,
                        "posthoc_source_fallback_count": posthoc_count,
                        "no_improvement_tie_count": tie_count,
                        "simulation_chain": LSO_SIMULATION_CHAIN_TEXT,
                        "source_artifacts": LSO_SOURCE_ARTIFACTS,
                    }
                )
            row_index += 1
    retention_totals = {
        method: {
            field: sum(int(row[field]) for row in csv_rows if row["method"] == method)
            for field in ("source_retained_count", "embedded_stored_source_count", "posthoc_source_fallback_count", "no_improvement_tie_count")
        }
        for method in METHODS
    }
    require(
        retention_totals
        == {
            "latent_ea": {"source_retained_count": 83, "embedded_stored_source_count": 83, "posthoc_source_fallback_count": 0, "no_improvement_tie_count": 83},
            "cem": {"source_retained_count": 132, "embedded_stored_source_count": 0, "posthoc_source_fallback_count": 131, "no_improvement_tie_count": 1},
            "cmaes": {"source_retained_count": 85, "embedded_stored_source_count": 0, "posthoc_source_fallback_count": 85, "no_improvement_tie_count": 0},
        },
        f"Unexpected source-retention totals: {retention_totals}",
    )
    csv_path = write_csv(
        "lso_bucket_mean_improvement.csv",
        ["metric_policy", "retained_improvement_definition", "model", "model_name", "method", "method_name", "bucket", "bucket_min", "bucket_max_finite", "dataset_exact_max_metadata", "n", "mean_retained_score", "sample_sd_retained_score", "max_retained_score", "mean_retained_improvement", "sample_sd_retained_improvement", "source_retained_count", "embedded_stored_source_count", "posthoc_source_fallback_count", "no_improvement_tie_count", "simulation_chain", "source_artifacts"],
        csv_rows,
    )
    cmap = LinearSegmentedColormap.from_list("improvement", ["#FFF7EC", "#FDBB84", "#D94701"])
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    image = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=max(1.45, float(data.max())))
    ax.grid(False)
    ax.set_xticks(range(6), [bucket_label(bucket, multiline=True) for bucket in BUCKETS])
    ax.set_yticks(range(15), row_labels)
    ax.set_title("Mean retained improvement by source-fitness bucket")
    for i in range(15):
        for j in range(6):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=6.8, color="white" if data[i, j] > 0.82 else "#222222")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, label="Mean retained improvement")
    fig.tight_layout()
    return [save_figure(fig, "lso_bucket_mean_improvement.eps"), csv_path]


def load_mini(
    manifest: list[dict[str, str]], dataset: list[tuple[str, str, float]]
) -> tuple[list[dict[str, str]], dict[tuple[str, int, int], dict[str, str]]]:
    with MINI.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 150, f"Mini endpoints have {len(rows)} rows, expected 150")
    manifest_keys = {(row["bucket_id"], int(row["seed_rank"]), int(row["dataset_idx"])): row for row in manifest}
    lookup: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        require(row["method"] == "frams", "Mini endpoint method is not frams")
        key = key_of(row)
        require(key in manifest_keys and key not in lookup, f"Invalid or duplicate mini key {key}")
        source = manifest_keys[key]
        require(row["seed_genotype"] == source["genotype"], f"Mini genotype mismatch for {key}")
        require(close(finite_float(row["seed_source_fitness"], "mini source fitness"), finite_float(source["fitness"], "manifest fitness")), f"Mini source fitness mismatch for {key}")
        retained(row)
        lookup[key] = row
    require(set(lookup) == set(manifest_keys), "Mini keys do not exactly match manifest")
    return rows, lookup


def selected_lso_lookup(endpoint_lookup: dict[tuple[str, int, int, str, str], dict[str, str]]) -> dict[str, dict[tuple[str, int, int], dict[str, str]]]:
    return {
        "tree_lso": {key[:3]: row for key, row in endpoint_lookup.items() if key[3:] == ("09_tree_vae_epoch80", "latent_ea")},
        "transformer_lso": {key[:3]: row for key, row in endpoint_lookup.items() if key[3:] == ("10_transformer_vae_epoch90", "latent_ea")},
    }


def generate_mini_endpoint_figures(
    endpoint_lookup: dict[tuple[str, int, int, str, str], dict[str, str]],
    mini_rows: list[dict[str, str]],
    mini_lookup: dict[tuple[str, int, int], dict[str, str]],
) -> list[Path]:
    lso = selected_lso_lookup(endpoint_lookup)
    series_rows = {
        "tree_lso": list(lso["tree_lso"].values()),
        "transformer_lso": list(lso["transformer_lso"].values()),
        "mini_ea": mini_rows,
    }
    aggregate_rows: list[dict[str, object]] = []
    means, errors = [], []
    for series, name in SERIES:
        scores = [retained(row)[0] for row in series_rows[series]]
        require(len(scores) == 150, f"Series {series} has {len(scores)} endpoints")
        means.append(statistics.mean(scores))
        errors.append(sample_sd(scores))
        aggregate_rows.append({"series": series, "series_name": name, "n": 150, "mean_retained_final_score": fcsv(means[-1]), "sample_sd_retained_final_score": fcsv(errors[-1]), "error_bar": "full sample SD", "retained_policy": "max(best_score,seed_source_fitness)", "lso_simulation_chain": LSO_SIMULATION_CHAIN_TEXT, "mini_simulation_chain": MINI_SIMULATION_CHAIN_TEXT, "lso_source_artifacts": LSO_SOURCE_ARTIFACTS, "mini_source_artifacts": MINI_SOURCE_ARTIFACTS, "comparison_scope": COMPARISON_SCOPE})
    aggregate_csv = write_csv("lso_mini_aggregate.csv", ["series", "series_name", "n", "mean_retained_final_score", "sample_sd_retained_final_score", "error_bar", "retained_policy", "lso_simulation_chain", "mini_simulation_chain", "lso_source_artifacts", "mini_source_artifacts", "comparison_scope"], aggregate_rows)
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    x = np.arange(3)
    for index, (series, _) in enumerate(SERIES):
        ax.errorbar(index, means[index], yerr=errors[index], fmt="o", color=SERIES_COLORS[series], capsize=5, markersize=5, linewidth=1.2)
    ax.set_xticks(x, [name.replace(" + ", "\n+ ") for _, name in SERIES])
    ax.set_ylabel("Retained final score")
    ax.set_title("Fixed-method retained-score estimates")
    ax.set_xlim(-0.55, 2.55)
    aggregate_eps = save_figure(fig, "lso_mini_aggregate.eps")

    aligned_rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    for col, series in enumerate(("tree_lso", "transformer_lso")):
        name = dict(SERIES)[series]
        keys = sorted(mini_lookup)
        x_values, y_values, differences = [], [], []
        for key in keys:
            mini_row = mini_lookup[key]
            lso_row = lso[series][key]
            require(mini_row["seed_genotype"] == lso_row["seed_genotype"], f"Aligned genotype mismatch for {key}")
            require(close(finite_float(mini_row["seed_source_fitness"], "mini source"), finite_float(lso_row["seed_source_fitness"], "LSO source")), f"Aligned fitness mismatch for {key}")
            mini_score = retained(mini_row)[0]
            lso_score = retained(lso_row)[0]
            difference = lso_score - mini_score
            x_values.append(mini_score)
            y_values.append(lso_score)
            differences.append(difference)
        diff_mean, diff_sd = statistics.mean(differences), sample_sd(differences)
        for key, mini_score, lso_score, difference in zip(keys, x_values, y_values, differences):
            aligned_rows.append({"lso_series": series, "lso_series_name": name, "bucket_id": key[0], "seed_rank": key[1], "seed_dataset_idx": key[2], "mini_retained_score": fcsv(mini_score), "lso_retained_score": fcsv(lso_score), "lso_minus_mini": fcsv(difference), "group_n": 150, "mean_lso_minus_mini": fcsv(diff_mean), "sample_sd_lso_minus_mini": fcsv(diff_sd), "join_key": "bucket_id,seed_rank,seed_dataset_idx", "selection_policy": "fixed configuration; no per-seed selection", "lso_simulation_chain": LSO_SIMULATION_CHAIN_TEXT, "mini_simulation_chain": MINI_SIMULATION_CHAIN_TEXT, "lso_source_artifacts": LSO_SOURCE_ARTIFACTS, "mini_source_artifacts": MINI_SOURCE_ARTIFACTS, "comparison_scope": COMPARISON_SCOPE})
        low = min(x_values + y_values)
        high = max(x_values + y_values)
        axes[0, col].scatter(x_values, y_values, s=12, color=SERIES_COLORS[series], edgecolors="white", linewidths=0.25)
        axes[0, col].plot([low, high], [low, high], color="#555555", linestyle="--", linewidth=0.9)
        axes[0, col].set_title(name)
        axes[0, col].set_xlabel("Mini retained score")
        axes[0, col].set_ylabel("LSO retained score")
        axes[1, col].hist(differences, bins=18, color=lighten(SERIES_COLORS[series], 0.45), edgecolor=SERIES_COLORS[series], linewidth=0.5)
        axes[1, col].axvline(0, color="#555555", linestyle="--", linewidth=0.9)
        axes[1, col].set_xlabel("Aligned LSO minus mini score")
        axes[1, col].set_ylabel("Sources")
    fig.suptitle("Aligned fixed-configuration comparison", y=1.01)
    fig.tight_layout()
    aligned_csv = write_csv("lso_mini_aligned_comparison.csv", ["lso_series", "lso_series_name", "bucket_id", "seed_rank", "seed_dataset_idx", "mini_retained_score", "lso_retained_score", "lso_minus_mini", "group_n", "mean_lso_minus_mini", "sample_sd_lso_minus_mini", "join_key", "selection_policy", "lso_simulation_chain", "mini_simulation_chain", "lso_source_artifacts", "mini_source_artifacts", "comparison_scope"], aligned_rows)
    aligned_eps = save_figure(fig, "lso_mini_aligned_comparison.eps")

    bucket_rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    centers = np.arange(6)
    offsets = (-0.18, 0.0, 0.18)
    for offset, (series, name) in zip(offsets, SERIES):
        means_by_bucket, sd_by_bucket = [], []
        for bucket in BUCKETS:
            part = [row for row in series_rows[series] if row["bucket_id"] == bucket]
            scores = [retained(row)[0] for row in part]
            require(len(scores) == 25, f"Mini comparison cell {(series, bucket)} has {len(scores)} rows")
            mean, sd = statistics.mean(scores), sample_sd(scores)
            means_by_bucket.append(mean)
            sd_by_bucket.append(sd)
            bucket_rows.append({"series": series, "series_name": name, "bucket_id": bucket, "bucket_min": fcsv(BUCKET_LIMITS[bucket][0]), "bucket_max_finite": fcsv(BUCKET_LIMITS[bucket][1]), "n": 25, "mean_retained_final_score": fcsv(mean), "sample_sd_retained_final_score": fcsv(sd), "error_bar": "full sample SD", "retained_policy": "max(best_score,seed_source_fitness)", "lso_simulation_chain": LSO_SIMULATION_CHAIN_TEXT, "mini_simulation_chain": MINI_SIMULATION_CHAIN_TEXT, "lso_source_artifacts": LSO_SOURCE_ARTIFACTS, "mini_source_artifacts": MINI_SOURCE_ARTIFACTS, "comparison_scope": COMPARISON_SCOPE})
        ax.errorbar(centers + offset, means_by_bucket, yerr=sd_by_bucket, fmt="o", color=SERIES_COLORS[series], capsize=2.5, markersize=4, linewidth=1.0, label=name)
    ax.set_xticks(centers, [bucket_label(bucket, multiline=True) for bucket in BUCKETS])
    ax.set_ylabel("Mean retained final score")
    ax.set_xlabel("Source-fitness bucket")
    ax.set_title("Fixed methods by source-fitness bucket")
    ax.legend(loc="upper left")
    fig.tight_layout()
    bucket_csv = write_csv("lso_mini_bucket_comparison.csv", ["series", "series_name", "bucket_id", "bucket_min", "bucket_max_finite", "n", "mean_retained_final_score", "sample_sd_retained_final_score", "error_bar", "retained_policy", "lso_simulation_chain", "mini_simulation_chain", "lso_source_artifacts", "mini_source_artifacts", "comparison_scope"], bucket_rows)
    bucket_eps = save_figure(fig, "lso_mini_bucket_comparison.eps")
    return [aggregate_eps, aggregate_csv, aligned_eps, aligned_csv, bucket_eps, bucket_csv]


def load_mini_history(mini_lookup: dict[tuple[str, int, int], dict[str, str]]) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    states: dict[tuple[str, int, int], tuple[int, float, int]] = {}
    count = 0
    with MINI_HISTORY.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            count += 1
            require(row["method"] == "frams", "Unexpected mini history method")
            key = key_of(row)
            require(key in mini_lookup, f"Unknown mini history key {key}")
            step = int(row["step"])
            slots = int(row["evaluations_requested"])
            require(slots == step * 100, f"Mini requested slots mismatch for {key}")
            score = max(finite_float(row["best_score"], "mini history best"), finite_float(row["seed_source_fitness"], "mini history source"))
            require(close(score, finite_float(row["best_or_source_score"], "mini history retained")), f"Mini history retained mismatch for {key}")
            previous = states.get(key)
            if previous is None:
                require(step == 1, f"Mini history for {key} does not start at one")
            else:
                require(step == previous[0] + 1, f"Non-contiguous mini history for {key}")
                require(score + 1e-10 >= previous[1], f"Mini retained score decreases for {key}")
            states[key] = (step, score, slots)
            values[slots].append(score)
    require(count == 45_000, f"Mini history has {count} rows, expected 45000")
    require(len(states) == 150 and len(values) == 300, "Mini history shape mismatch")
    for key, (step, score, slots) in states.items():
        require(step == 300 and slots == 30_000, f"Mini history endpoint missing for {key}")
        require(close(score, retained(mini_lookup[key])[0]), f"Mini history endpoint differs from final row for {key}")
    require(all(len(part) == 150 for part in values.values()), "Mini history slot count mismatch")
    return values


def generate_mini_convergence(
    lso_history: dict[tuple[str, str, int], list[float]],
    mini_history: dict[int, list[float]],
) -> list[Path]:
    selected = {
        "tree_lso": {slot: lso_history[("09_tree_vae_epoch80", "latent_ea", slot)] for slot in range(100, 30_001, 100)},
        "transformer_lso": {slot: lso_history[("10_transformer_vae_epoch90", "latent_ea", slot)] for slot in range(100, 30_001, 100)},
        "mini_ea": mini_history,
    }
    selected_lso_count = sum(len(v) for key in ("tree_lso", "transformer_lso") for v in selected[key].values())
    selected_mini_count = sum(len(v) for v in selected["mini_ea"].values())
    require(selected_lso_count == 90_000, f"Selected LSO history has {selected_lso_count} rows, expected 90000")
    require(selected_mini_count == 45_000, f"Mini history has {selected_mini_count} rows, expected 45000")
    rows: list[dict[str, object]] = []
    x = np.arange(100, 30_001, 100)
    fig, axes = plt.subplots(1, 3, figsize=(7.7, 3.1), sharex=True, sharey=True)
    for ax, (series, name) in zip(axes, SERIES):
        summaries = []
        for slot in x:
            scores = selected[series][int(slot)]
            require(len(scores) == 150, f"Selected convergence group {(series, slot)} has {len(scores)} rows")
            q1, median, q3 = quantiles(scores)
            summaries.append((q1, median, q3))
            rows.append({"series": series, "series_name": name, "requested_candidate_slots": int(slot), "n": 150, "q1_retained_score": fcsv(q1), "median_retained_score": fcsv(median), "q3_retained_score": fcsv(q3), "metric": "retained_best_so_far=max(best_score,seed_source_fitness)", "lso_simulation_chain": LSO_SIMULATION_CHAIN_TEXT, "mini_simulation_chain": MINI_SIMULATION_CHAIN_TEXT, "lso_source_artifacts": LSO_SOURCE_ARTIFACTS, "mini_source_artifacts": MINI_SOURCE_ARTIFACTS, "comparison_scope": COMPARISON_SCOPE})
        color = SERIES_COLORS[series]
        ax.fill_between(x, [item[0] for item in summaries], [item[2] for item in summaries], color=lighten(color), linewidth=0)
        ax.plot(x, [item[1] for item in summaries], color=color)
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel("Requested candidate slots")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    axes[0].set_ylabel("Retained best-so-far score")
    fig.suptitle("Fixed-method convergence", y=1.01)
    fig.tight_layout()
    csv_path = write_csv("lso_mini_convergence.csv", ["series", "series_name", "requested_candidate_slots", "n", "q1_retained_score", "median_retained_score", "q3_retained_score", "metric", "lso_simulation_chain", "mini_simulation_chain", "lso_source_artifacts", "mini_source_artifacts", "comparison_scope"], rows)
    return [save_figure(fig, "lso_mini_convergence.eps"), csv_path]


def latex_escape(text: str) -> str:
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(char, char) for char in text)


def breakable_genotype(genotype: str) -> str:
    escaped = latex_escape(genotype)
    chunks = [escaped[index : index + 10] for index in range(0, len(escaped), 10)]
    return r"\texttt{" + r"\allowbreak{}".join(chunks) + "}"


def generate_representatives(
    manifest: list[dict[str, str]],
    endpoint_lookup: dict[tuple[str, int, int, str, str], dict[str, str]],
    mini_lookup: dict[tuple[str, int, int], dict[str, str]],
) -> list[Path]:
    lso = selected_lso_lookup(endpoint_lookup)
    header = "Bkt. & Index & Outcome & Genotype & Reported score & Source retained \\\\"
    lines = [
        "% Generated by scripts/generate_thesis_lso_figures.py; do not edit manually.",
        "\\begingroup",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\setlength{\\LTcapwidth}{\\textwidth}",
        "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.045\\textwidth}>{\\raggedright\\arraybackslash}p{0.055\\textwidth}>{\\raggedright\\arraybackslash}p{0.15\\textwidth}>{\\raggedright\\arraybackslash}p{0.37\\textwidth}>{\\raggedright\\arraybackslash}p{0.09\\textwidth}>{\\raggedright\\arraybackslash}p{0.07\\textwidth}@{}}",
        "\\caption{Representative source and retained optimization outcomes. Source rows report stored manifest fitness; optimization rows report $\\max(F^{\\mathrm{raw}},F^{\\mathrm{source}})$. The final column states whether this retained endpoint kept the source genotype.}",
        "\\label{tab:representative-genotype-outcomes}\\\\",
        "\\hline",
        header,
        "\\hline",
        "\\endfirsthead",
        "\\multicolumn{6}{c}{\\tablename\\ \\thetable{} -- continued}\\\\",
        "\\hline",
        header,
        "\\hline",
        "\\endhead",
        "\\hline",
        "\\multicolumn{6}{r}{Continued on next page}\\\\",
        "\\endfoot",
        "\\hline",
        "\\endlastfoot",
    ]
    for bucket in BUCKETS:
        candidates = [row for row in manifest if row["bucket_id"] == bucket]
        median = statistics.median(finite_float(row["fitness"], "manifest fitness") for row in candidates)
        chosen = min(candidates, key=lambda row: (abs(finite_float(row["fitness"], "manifest fitness") - median), int(row["dataset_idx"])))
        key = (bucket, int(chosen["seed_rank"]), int(chosen["dataset_idx"]))
        outcomes = [("Source", chosen["genotype"], finite_float(chosen["fitness"], "manifest fitness"), None)]
        for name, row in (
            ("TreeVAE-80 EA", lso["tree_lso"][key]),
            ("TransVAE-90 EA", lso["transformer_lso"][key]),
            ("Mini EA", mini_lookup[key]),
        ):
            score, _, source_retained, genotype = retained(row)
            outcomes.append((name, genotype, score, source_retained))
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textbf{{{bucket_label(bucket)}}}}}\\\\*")
        for name, genotype, score, source_retained in outcomes:
            retained_label = "--" if source_retained is None else ("yes" if source_retained else "no")
            lines.append(f"{bucket} & {chosen['dataset_idx']} & {latex_escape(name)} & {breakable_genotype(genotype)} & {score:.6f} & {retained_label} \\\\")
        if bucket != BUCKETS[-1]:
            lines.append("\\noalign{\\vskip 2pt}")
    lines.extend(["\\end{longtable}", "\\endgroup", ""])
    path = FIGURES / "representative_genotype_outcomes.tex"
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return [path]


def generate_phenotype(path: Path) -> list[Path]:
    require(path.is_file(), f"Phenotype image does not exist: {path}")
    image = plt.imread(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.0, 1.35]})
    axes[0].axis("off")
    axes[0].set_title("F1 genotype")
    axes[0].text(0.5, 0.5, "\n".join(textwrap.wrap(PHENOTYPE_GENOTYPE, width=22, break_long_words=True, break_on_hyphens=False)), ha="center", va="center", family="monospace", fontsize=11, bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F4F4F4", "edgecolor": "#777777"})
    axes[1].imshow(image)
    axes[1].axis("off")
    axes[1].set_title("Rendered phenotype")
    fig.tight_layout()
    eps_path = save_figure(fig, "f1_genotype_phenotype.eps")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = write_csv(
        "f1_genotype_phenotype.csv",
        ["genotype", "source_image_name", "source_image_sha256", "association"],
        [{"genotype": PHENOTYPE_GENOTYPE, "source_image_name": path.name, "source_image_sha256": digest, "association": "declared explicitly through generator CLI"}],
    )
    return [eps_path, metadata_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenotype-image", type=Path, help="Raster phenotype image for the optional genotype/phenotype composition")
    parser.add_argument("--phenotype-genotype", help="Genotype explicitly associated with --phenotype-image")
    args = parser.parse_args()
    require(
        (args.phenotype_image is None) == (args.phenotype_genotype is None),
        "--phenotype-image and --phenotype-genotype must be supplied together",
    )
    if args.phenotype_genotype is not None:
        require(
            normalized_genotype(args.phenotype_genotype) == normalized_genotype(PHENOTYPE_GENOTYPE),
            "The supplied phenotype genotype does not match the documented figure genotype",
        )
    configure_style()
    require(FIGURES.is_dir(), f"Figure output directory does not exist: {FIGURES}")
    created: list[Path] = []
    dataset = load_dataset()
    manifest = load_manifest(dataset)
    validate_simulation_provenance()
    created.extend(generate_pipeline())
    created.extend(generate_dataset_figure(dataset, manifest))
    created.extend(generate_training_curves())
    endpoint_rows, endpoint_lookup = load_endpoints(manifest)
    created.extend(generate_matrix_and_distributions(endpoint_rows))
    convergence_files, lso_history = generate_convergence(endpoint_lookup)
    created.extend(convergence_files)
    created.extend(generate_bucket_improvement(endpoint_rows))
    mini_rows, mini_lookup = load_mini(manifest, dataset)
    created.extend(generate_mini_endpoint_figures(endpoint_lookup, mini_rows, mini_lookup))
    mini_history = load_mini_history(mini_lookup)
    created.extend(generate_mini_convergence(lso_history, mini_history))
    created.extend(generate_representatives(manifest, endpoint_lookup, mini_lookup))
    if args.phenotype_image is None:
        print("SKIPPED f1_genotype_phenotype.eps: phenotype image and genotype were not supplied")
    else:
        created.extend(generate_phenotype(args.phenotype_image.resolve()))
    print(f"VALIDATED dataset={len(dataset)} max={max(row[2] for row in dataset)!r} manifest={len(manifest)}")
    print("VALIDATED LSO endpoints=2250 groups=15 rows_per_group=150 rows_per_bucket=25")
    print("VALIDATED LSO histories=675000 steps_per_source=300 selected_LSO_histories=90000")
    print("VALIDATED mini endpoints=150 mini_histories=45000 aligned_keys=150")
    print(f"VALIDATED simulation chains LSO={LSO_SIMULATION_CHAIN_TEXT} mini={MINI_SIMULATION_CHAIN_TEXT}")
    print(f"CREATED {len(created)} files")
    for path in created:
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
