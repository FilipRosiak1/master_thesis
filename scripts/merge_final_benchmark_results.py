from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge per-job CSV files from the final seed-bucket benchmark")
    parser.add_argument("run_dir")
    parser.add_argument("--job-plan", default=None, help="Defaults to <run_dir>/job_plan.csv")
    parser.add_argument("--output", default=None, help="Defaults to <run_dir>/final_benchmark.csv")
    parser.add_argument("--history-output", default=None, help="Defaults to <run_dir>/final_benchmark.history.csv")
    return parser


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _plan_relative(path: str, run_dir: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else run_dir / value


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = _resolve(args.run_dir)
    job_plan = _resolve(args.job_plan) if args.job_plan else run_dir / "job_plan.csv"
    output = _resolve(args.output) if args.output else run_dir / "final_benchmark.csv"
    history_output = _resolve(args.history_output) if args.history_output else run_dir / "final_benchmark.history.csv"

    rows: list[dict[str, str]] = []
    history_rows: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []
    for plan_row in _read_csv(job_plan):
        csv_path = _plan_relative(plan_row["output_csv"], run_dir)
        history_path = _plan_relative(plan_row["output_history"], run_dir)
        if csv_path.exists():
            rows.extend(_read_csv(csv_path))
        else:
            missing.append({**plan_row, "missing_path": str(csv_path)})
        if history_path.exists():
            history_rows.extend(_read_csv(history_path))

    _write_csv(output, rows)
    _write_csv(history_output, history_rows)
    summary = {
        "run_dir": str(run_dir),
        "job_plan": str(job_plan),
        "rows": len(rows),
        "history_rows": len(history_rows),
        "missing_jobs": len(missing),
        "missing": missing,
    }
    summary_path = output.with_suffix(".merge_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {output}")
    print(f"Wrote: {history_output}")
    print(f"Wrote: {summary_path}")
    if missing:
        print(f"Missing job outputs: {len(missing)}")


if __name__ == "__main__":
    main()
