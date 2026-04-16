from __future__ import annotations

from pathlib import Path


def parse_gen_file(path: Path) -> tuple[str, str] | None:
    genotype = None
    fitness = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("genotype:"):
                genotype = line.split("genotype:", 1)[1].strip()
                if (
                    genotype.startswith("~")
                    and genotype.endswith("~")
                    and len(genotype) >= 2
                ):
                    genotype = genotype[1:-1].strip()
            elif line.startswith("vertpos:"):
                fitness = line.split("vertpos:", 1)[1].strip()

    if not genotype or fitness is None:
        return None

    return genotype, fitness


def build_dataset(raw_root: Path, output_file: Path) -> tuple[int, int]:
    rows: list[str] = []
    seen_rows: set[str] = set()
    skipped = 0

    # Sort by full path to keep deterministic output.
    for gen_file in sorted(raw_root.rglob("*.gen"), key=lambda p: str(p)):
        parsed = parse_gen_file(gen_file)
        if parsed is None:
            skipped += 1
            continue

        genotype, fitness = parsed
        row = f"{genotype}\t{fitness}"
        # Deduplicate exact genotype+fitness pairs while preserving first-seen order.
        if row not in seen_rows:
            seen_rows.add(row)
            rows.append(row)

    output_file.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows), skipped


def main() -> None:
    root = Path(__file__).resolve().parent
    raw_root = root / "raw_data"
    output_file = root / "f1_dataset.txt"

    count, skipped = build_dataset(raw_root, output_file)
    print(f"Saved {count} rows to {output_file}")
    if skipped:
        print(f"Skipped {skipped} files without both genotype and vertpos")


if __name__ == "__main__":
    main()
