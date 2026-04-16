from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import math


SOURCE = Path("datasets/f1/f1_dataset.txt")
OUT_EVEN = Path("datasets/f1/f1_dataset_1k_even.txt")
OUT_SAME = Path("datasets/f1/f1_dataset_1k_same.txt")

TARGET_SIZE = 1000
BIN_WIDTH = 0.1
SEED = 42


def parse_rows(path: Path) -> list[tuple[str, float, str]]:
    rows: list[tuple[str, float, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split("\t", 1)
        if len(parts) != 2:
            continue
        genotype, fitness_raw = parts
        try:
            fitness = float(fitness_raw)
        except ValueError:
            continue
        rows.append((genotype, fitness, row))
    return rows


def bin_index(value: float, min_edge: float) -> int:
    return int(math.floor((value - min_edge) / BIN_WIDTH))


def build_bins(rows: list[tuple[str, float, str]]):
    min_fit = min(r[1] for r in rows)
    max_fit = max(r[1] for r in rows)
    min_edge = math.floor(min_fit / BIN_WIDTH) * BIN_WIDTH
    max_edge = math.ceil(max_fit / BIN_WIDTH) * BIN_WIDTH
    n_bins = int(round((max_edge - min_edge) / BIN_WIDTH))

    bins: dict[int, list[tuple[str, float, str]]] = defaultdict(list)
    for row in rows:
        idx = bin_index(row[1], min_edge)
        if idx >= n_bins:
            idx = n_bins - 1
        bins[idx].append(row)

    non_empty = sorted([i for i in range(n_bins) if bins.get(i)])
    return bins, non_empty


def allocate_same_distribution(
    bin_counts: dict[int, int], total: int
) -> dict[int, int]:
    all_bins = sorted(bin_counts)
    total_count = sum(bin_counts.values())

    raw = {b: (bin_counts[b] / total_count) * total for b in all_bins}
    alloc = {b: int(math.floor(raw[b])) for b in all_bins}
    remainder = total - sum(alloc.values())

    frac_order = sorted(all_bins, key=lambda b: raw[b] - alloc[b], reverse=True)
    i = 0
    while remainder > 0:
        b = frac_order[i % len(frac_order)]
        alloc[b] += 1
        remainder -= 1
        i += 1
    return alloc


def allocate_even_as_possible(bin_counts: dict[int, int], total: int) -> dict[int, int]:
    bins = sorted(bin_counts)
    k = len(bins)
    base = total // k

    alloc = {b: min(base, bin_counts[b]) for b in bins}
    remainder = total - sum(alloc.values())

    # Water-filling: repeatedly add one item to the currently smallest allocated bin
    # that still has capacity. This gives the most even feasible distribution.
    while remainder > 0:
        candidates = [b for b in bins if alloc[b] < bin_counts[b]]
        if not candidates:
            break
        b = min(candidates, key=lambda x: (alloc[x], x))
        alloc[b] += 1
        remainder -= 1

    if sum(alloc.values()) != total:
        raise RuntimeError("Could not allocate requested sample size for even dataset")
    return alloc


def sample_by_allocation(
    bins: dict[int, list[tuple[str, float, str]]],
    allocation: dict[int, int],
    rng: random.Random,
) -> list[str]:
    sampled_rows: list[str] = []
    for b, n in allocation.items():
        if n == 0:
            continue
        sampled = rng.sample(bins[b], n)
        sampled_rows.extend([row[2] for row in sampled])
    rng.shuffle(sampled_rows)
    return sampled_rows


def write_rows(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    rows = parse_rows(SOURCE)
    if len(rows) < TARGET_SIZE:
        raise RuntimeError(f"Source dataset too small: {len(rows)} rows")

    bins, non_empty_bins = build_bins(rows)
    bin_counts = {b: len(bins[b]) for b in non_empty_bins}

    rng_same = random.Random(SEED)
    rng_even = random.Random(SEED + 1)

    alloc_same = allocate_same_distribution(bin_counts, TARGET_SIZE)
    alloc_even = allocate_even_as_possible(bin_counts, TARGET_SIZE)

    sampled_same = sample_by_allocation(bins, alloc_same, rng_same)
    sampled_even = sample_by_allocation(bins, alloc_even, rng_even)

    if len(sampled_same) != TARGET_SIZE or len(sampled_even) != TARGET_SIZE:
        raise RuntimeError("Wrong output size")

    write_rows(OUT_SAME, sampled_same)
    write_rows(OUT_EVEN, sampled_even)

    print(f"Saved {len(sampled_same)} rows to {OUT_SAME}")
    print(f"Saved {len(sampled_even)} rows to {OUT_EVEN}")


if __name__ == "__main__":
    main()
