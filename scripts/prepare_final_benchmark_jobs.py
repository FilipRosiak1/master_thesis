from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LABELS = (
    "09_tree_vae_epoch80",
    "10_transformer_vae_epoch90",
    "02_lhs_latent256_seed42_best",
    "06_lhs_depth_seed42_best",
    "exports/cluster_export_20260524_222538/models/f1_guided/fitness_aware_lhs_seed42",
)
DEFAULT_METHODS = ("latent_ea", "cem", "cmaes")
DEFAULT_BUCKETS = (
    ("B1", 0.30, 0.50),
    ("B2", 0.50, 0.75),
    ("B3", 0.75, 1.00),
    ("B4", 1.00, 1.20),
    ("B5", 1.20, 1.50),
    ("B6", 1.50, None),
)


def _build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Prepare seed manifest and Slurm job plans for the final F1 benchmark")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--checkpoints-root", default="exports/f1_selected_10_ckpts_20260520_184738")
    parser.add_argument("--output-dir", default=f"exports/final_seed_bucket_benchmark_{stamp}")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS), help="Comma-separated AE checkpoint labels")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated latent methods: latent_ea,cem,cmaes")
    parser.add_argument("--seed-count-per-bucket", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=20260607)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--no-baseline", action="store_true", help="Do not add CPU Framsticks baseline jobs")
    parser.add_argument("--allow-short-buckets", action="store_true", help="Use all matching seeds if a bucket has fewer than requested")
    return parser


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_name(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value]
    return "".join(chars).strip("_") or "item"


def _label_slug(label: str) -> str:
    normalized = label.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized


def _label_exists(label: str, checkpoints_root: Path) -> bool:
    candidate = checkpoints_root / label
    if candidate.exists():
        return True
    value = Path(label)
    if value.is_absolute():
        return value.exists()
    return (ROOT / value).exists()


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.8g}"


def _parse_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw in enumerate(handle):
            raw = raw.strip()
            if not raw:
                continue
            if "\t" in raw:
                genotype, fitness_raw = raw.split("\t", 1)
            else:
                parts = raw.split()
                if len(parts) < 2:
                    continue
                genotype, fitness_raw = parts[0], parts[-1]
            genotype = "".join(genotype.split())
            try:
                fitness = float(fitness_raw.strip())
            except ValueError:
                continue
            if not genotype or math.isnan(fitness) or math.isinf(fitness):
                continue
            rows.append({"dataset_idx": idx, "genotype": genotype, "fitness": fitness})
    return rows


def _in_bucket(value: float, lower: float, upper: float | None) -> bool:
    if value < lower:
        return False
    return upper is None or value < upper


def _sample_manifest(
    dataset_rows: list[dict[str, Any]],
    *,
    seed_count: int,
    random_seed: int,
    allow_short_buckets: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(random_seed)
    manifest: list[dict[str, Any]] = []
    for bucket_id, lower, upper in DEFAULT_BUCKETS:
        candidates = [row for row in dataset_rows if _in_bucket(float(row["fitness"]), lower, upper)]
        if len(candidates) < seed_count and not allow_short_buckets:
            raise RuntimeError(
                f"Bucket {bucket_id} has only {len(candidates)} candidates, requested {seed_count}. "
                "Use --allow-short-buckets to continue."
            )
        selected_count = min(seed_count, len(candidates))
        selected = rng.sample(candidates, selected_count)
        for rank, row in enumerate(selected, start=1):
            manifest.append(
                {
                    "bucket_id": bucket_id,
                    "bucket_min": _format_float(lower),
                    "bucket_max": _format_float(upper),
                    "seed_rank": rank,
                    "dataset_idx": row["dataset_idx"],
                    "genotype": row["genotype"],
                    "fitness": _format_float(float(row["fitness"])),
                    "sampling_seed": random_seed,
                }
            )
    return manifest


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _bucket_ids(manifest: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in manifest:
        bucket = str(row["bucket_id"])
        if bucket not in seen:
            seen.append(bucket)
    return seen


def _make_output_paths(job_type: str, global_job_id: int, method: str, bucket_id: str, label: str = "") -> dict[str, str]:
    label_part = f"_{_safe_name(_label_slug(label))}" if label else ""
    stem = f"job_{global_job_id:04d}{label_part}_{_safe_name(method)}_{bucket_id}"
    subdir = "gpu" if job_type == "latent" else "cpu"
    return {
        "output_csv": f"jobs/{subdir}/{stem}.csv",
        "output_history": f"jobs/{subdir}/{stem}.history.csv",
        "output_details": f"jobs/{subdir}/{stem}.details.json",
    }


def _make_plan(
    *,
    labels: list[str],
    methods: list[str],
    bucket_ids: list[str],
    seed_count: int,
    generations: int,
    iterations: int,
    population_size: int,
    include_baseline: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    global_job_id = 0
    gpu_rows: list[dict[str, Any]] = []
    cpu_rows: list[dict[str, Any]] = []

    for label in labels:
        for method in methods:
            for bucket_id in bucket_ids:
                row = {
                    "global_job_id": global_job_id,
                    "job_type": "latent",
                    "model_label": label,
                    "method": method,
                    "bucket_id": bucket_id,
                    "seed_count": seed_count,
                    "generations": generations,
                    "iterations": iterations,
                    "population_size": population_size,
                }
                row.update(_make_output_paths("latent", global_job_id, method, bucket_id, label))
                gpu_rows.append(row)
                global_job_id += 1

    if include_baseline:
        for bucket_id in bucket_ids:
            row = {
                "global_job_id": global_job_id,
                "job_type": "baseline",
                "model_label": "",
                "method": "frams",
                "bucket_id": bucket_id,
                "seed_count": seed_count,
                "generations": generations,
                "iterations": "",
                "population_size": population_size,
            }
            row.update(_make_output_paths("baseline", global_job_id, "frams", bucket_id))
            cpu_rows.append(row)
            global_job_id += 1

    return gpu_rows + cpu_rows, gpu_rows, cpu_rows


def _write_plan(path: Path, rows: list[dict[str, Any]]) -> None:
    prepared = []
    for array_job_id, row in enumerate(rows):
        prepared.append({"array_job_id": array_job_id, **row})
    fieldnames = [
        "array_job_id",
        "global_job_id",
        "job_type",
        "model_label",
        "method",
        "bucket_id",
        "seed_count",
        "generations",
        "iterations",
        "population_size",
        "output_csv",
        "output_history",
        "output_details",
    ]
    _write_csv(path, prepared, fieldnames)


def main() -> None:
    args = _build_parser().parse_args()
    labels = _split_csv(args.labels)
    methods = _split_csv(args.methods)
    invalid_methods = sorted(set(methods) - set(DEFAULT_METHODS))
    if invalid_methods:
        raise ValueError(f"Unsupported methods: {', '.join(invalid_methods)}")
    if not labels:
        raise ValueError("At least one model label is required")
    if not methods:
        raise ValueError("At least one latent method is required")

    output_dir = _resolve(args.output_dir)
    checkpoints_root = _resolve(args.checkpoints_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobs" / "gpu").mkdir(parents=True, exist_ok=True)
    (output_dir / "jobs" / "cpu").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    data_path = _resolve(args.data_path)
    missing_labels = [label for label in labels if not _label_exists(label, checkpoints_root)]
    dataset_rows = _parse_dataset(data_path)
    manifest = _sample_manifest(
        dataset_rows,
        seed_count=args.seed_count_per_bucket,
        random_seed=args.random_seed,
        allow_short_buckets=args.allow_short_buckets,
    )
    bucket_ids = _bucket_ids(manifest)

    manifest_path = output_dir / "seed_manifest.csv"
    _write_csv(
        manifest_path,
        manifest,
        ["bucket_id", "bucket_min", "bucket_max", "seed_rank", "dataset_idx", "genotype", "fitness", "sampling_seed"],
    )

    all_rows, gpu_rows, cpu_rows = _make_plan(
        labels=labels,
        methods=methods,
        bucket_ids=bucket_ids,
        seed_count=args.seed_count_per_bucket,
        generations=args.generations,
        iterations=args.iterations,
        population_size=args.population_size,
        include_baseline=not args.no_baseline,
    )
    _write_plan(output_dir / "job_plan.csv", all_rows)
    _write_plan(output_dir / "job_plan.gpu.csv", gpu_rows)
    _write_plan(output_dir / "job_plan.cpu.csv", cpu_rows)

    summary = {
        "data_path": str(data_path),
        "checkpoints_root": str(checkpoints_root),
        "output_dir": str(output_dir),
        "labels": labels,
        "missing_labels": missing_labels,
        "methods": methods,
        "buckets": [
            {"bucket_id": bucket_id, "min": lower, "max": upper}
            for bucket_id, lower, upper in DEFAULT_BUCKETS
        ],
        "seed_count_per_bucket": args.seed_count_per_bucket,
        "total_seed_rows": len(manifest),
        "gpu_jobs": len(gpu_rows),
        "cpu_jobs": len(cpu_rows),
        "total_jobs": len(all_rows),
        "generations": args.generations,
        "iterations": args.iterations,
        "population_size": args.population_size,
        "random_seed": args.random_seed,
    }
    (output_dir / "benchmark_plan.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {output_dir / 'job_plan.csv'}")
    print(f"Wrote: {output_dir / 'job_plan.gpu.csv'} ({len(gpu_rows)} jobs)")
    print(f"Wrote: {output_dir / 'job_plan.cpu.csv'} ({len(cpu_rows)} jobs)")
    print(f"Wrote: {output_dir / 'benchmark_plan.summary.json'}")
    if missing_labels:
        print("WARNING: missing checkpoint labels under --checkpoints-root:")
        for label in missing_labels:
            print(f"  {label}")


if __name__ == "__main__":
    main()
