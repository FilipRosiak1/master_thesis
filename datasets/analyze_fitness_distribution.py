from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_fitnesses(dataset_path: Path) -> np.ndarray:
    values = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            parts = row.split("\t", 1)
            if len(parts) != 2:
                continue
            try:
                values.append(float(parts[1]))
            except ValueError:
                continue
    return np.array(values, dtype=float)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "count": float(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def make_bins(values: np.ndarray, bin_width: float) -> np.ndarray:
    lo = math.floor(float(np.min(values)) / bin_width) * bin_width
    hi = math.ceil(float(np.max(values)) / bin_width) * bin_width
    return np.arange(lo, hi + bin_width * 1.0001, bin_width)


def save_bucket_csv(edges: np.ndarray, counts: np.ndarray, out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket_start", "bucket_end", "count"])
        for i, count in enumerate(counts):
            writer.writerow([f"{edges[i]:.6f}", f"{edges[i + 1]:.6f}", int(count)])


def save_histogram(values: np.ndarray, edges: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.hist(values, bins=edges, edgecolor="black", linewidth=0.6)
    plt.title("F1 Dataset Fitness Distribution")
    plt.xlabel("Fitness (vertpos)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze fitness distribution in f1_dataset.txt"
    )
    parser.add_argument("--dataset", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--bin-width", type=float, default=0.1)
    parser.add_argument("--out-png", default="datasets/f1/fitness_histogram_0p1.png")
    parser.add_argument("--out-csv", default="datasets/f1/fitness_buckets_0p1.csv")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)

    values = load_fitnesses(dataset_path)
    if values.size == 0:
        raise RuntimeError(f"No valid fitness values found in {dataset_path}")

    edges = make_bins(values, args.bin_width)
    counts, _ = np.histogram(values, bins=edges)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    save_histogram(values, edges, out_png)
    save_bucket_csv(edges, counts, out_csv)

    stats = summarize(values)
    print("Fitness summary")
    for key in [
        "count",
        "min",
        "max",
        "mean",
        "std",
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    ]:
        print(f"  {key}: {stats[key]}")
    print(f"Saved histogram: {out_png}")
    print(f"Saved buckets  : {out_csv}")


if __name__ == "__main__":
    main()
