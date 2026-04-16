from __future__ import annotations

from pathlib import Path
import math

import matplotlib.pyplot as plt


BASE = Path("datasets/f1")
SRC = BASE / "f1_dataset.txt"
SAME = BASE / "f1_dataset_1k_same.txt"
EVEN = BASE / "f1_dataset_1k_even.txt"
OUT_OVERLAY = BASE / "fitness_distribution_1k_overlay.png"
OUT_PANELS = BASE / "fitness_distribution_1k_panels.png"


def load_fitness(path: Path) -> list[float]:
    vals: list[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split("\t", 1)
        if len(parts) != 2:
            continue
        try:
            vals.append(float(parts[1]))
        except ValueError:
            continue
    return vals


def main() -> None:
    src = load_fitness(SRC)
    same = load_fitness(SAME)
    even = load_fitness(EVEN)

    all_vals = src + same + even
    lo = math.floor(min(all_vals) * 10) / 10
    hi = math.ceil(max(all_vals) * 10) / 10
    bins = [lo + i * 0.1 for i in range(int(round((hi - lo) / 0.1)) + 1)]

    # Overlay (density, easier to compare shape)
    plt.figure(figsize=(12, 6))
    plt.hist(
        src,
        bins=bins,
        density=True,
        alpha=0.35,
        label="original (10893)",
        edgecolor="black",
        linewidth=0.4,
    )
    plt.hist(
        same,
        bins=bins,
        density=True,
        alpha=0.45,
        label="1k same",
        edgecolor="black",
        linewidth=0.4,
    )
    plt.hist(
        even,
        bins=bins,
        density=True,
        alpha=0.45,
        label="1k even",
        edgecolor="black",
        linewidth=0.4,
    )
    plt.title("Fitness Distribution Comparison (bin = 0.1)")
    plt.xlabel("fitness")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_OVERLAY, dpi=180)
    plt.close()

    # Side-by-side panels (counts)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    axes[0].hist(src, bins=bins, color="#4C78A8", edgecolor="black", linewidth=0.4)
    axes[0].set_title("original (10893)")
    axes[1].hist(same, bins=bins, color="#59A14F", edgecolor="black", linewidth=0.4)
    axes[1].set_title("1k same")
    axes[2].hist(even, bins=bins, color="#F28E2B", edgecolor="black", linewidth=0.4)
    axes[2].set_title("1k even")
    for ax in axes:
        ax.set_xlabel("fitness")
    axes[0].set_ylabel("count")
    fig.suptitle("Fitness Histograms (bin = 0.1)")
    fig.tight_layout()
    fig.savefig(OUT_PANELS, dpi=180)
    plt.close(fig)

    print(f"Saved: {OUT_OVERLAY}")
    print(f"Saved: {OUT_PANELS}")


if __name__ == "__main__":
    main()
