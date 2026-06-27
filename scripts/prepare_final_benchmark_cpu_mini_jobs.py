from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CANDIDATES = (
    ROOT / "exports/final_seed_bucket_benchmark_baseline_20260613_011516",
    ROOT / "exports/final_results_cpu/exports/final_seed_bucket_benchmark_baseline_20260613_011516",
)
MINI_SIM_CHAIN = ";".join(
    [
        "src/framsticks/framspy/eval-allcriteria-mini.sim",
        "src/framsticks/framspy/deterministic.sim",
        "src/framsticks/framspy/sample-period-2.sim",
        "src/framsticks/framspy/only-body.sim",
    ]
)


def _build_parser() -> argparse.ArgumentParser:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a CPU-only final baseline benchmark plan that reuses the exact seed manifest "
            "from the original CPU baseline run, but is intended for eval-allcriteria-mini.sim."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        default=None,
        help="Original CPU baseline run directory. Defaults to the known final baseline path if present.",
    )
    parser.add_argument(
        "--output-dir",
        default=f"exports/final_seed_bucket_benchmark_baseline_mini_{stamp}",
        help="New output directory for mini-sim CPU jobs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing plan files in an existing output directory.")
    return parser


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _default_source_run_dir() -> Path:
    for candidate in DEFAULT_SOURCE_CANDIDATES:
        if (candidate / "seed_manifest.csv").exists() and (candidate / "job_plan.cpu.csv").exists():
            return candidate
    checked = "\n".join(str(path) for path in DEFAULT_SOURCE_CANDIDATES)
    raise FileNotFoundError(f"Could not find default source run directory. Checked:\n{checked}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_required(source: Path, output: Path, filename: str, overwrite: bool) -> None:
    src = source / filename
    dst = output / filename
    if not src.exists():
        raise FileNotFoundError(f"Missing required source file: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {dst}. Use --overwrite if intended.")
    shutil.copy2(src, dst)


def main() -> int:
    args = _build_parser().parse_args()
    source_run_dir = _resolve(args.source_run_dir) if args.source_run_dir else _default_source_run_dir()
    output_dir = _resolve(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "jobs" / "cpu").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)

    _copy_required(source_run_dir, output_dir, "seed_manifest.csv", args.overwrite)
    _copy_required(source_run_dir, output_dir, "job_plan.cpu.csv", args.overwrite)

    cpu_rows = _read_csv(output_dir / "job_plan.cpu.csv")
    if not cpu_rows:
        raise RuntimeError(f"No CPU jobs found in {output_dir / 'job_plan.cpu.csv'}")
    fieldnames = list(cpu_rows[0].keys())
    job_plan_path = output_dir / "job_plan.csv"
    if job_plan_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {job_plan_path}. Use --overwrite if intended.")
    _write_csv(job_plan_path, cpu_rows, fieldnames)

    manifest_rows = _read_csv(output_dir / "seed_manifest.csv")
    summary = {
        "source_run_dir": str(source_run_dir),
        "output_dir": str(output_dir),
        "purpose": "CPU baseline rerun with eval-allcriteria-mini.sim and the original final-benchmark seeds",
        "framsticks_sim": MINI_SIM_CHAIN,
        "seed_manifest": "seed_manifest.csv",
        "job_plan_cpu": "job_plan.cpu.csv",
        "seed_rows": len(manifest_rows),
        "cpu_jobs": len(cpu_rows),
        "expected_rows": len(manifest_rows),
        "notes": [
            "This plan reuses the original seed_manifest.csv exactly; it does not resample seeds.",
            "Only CPU baseline jobs are prepared/submitted by the mini submit script.",
            "The mini simulation replaces eval-allcriteria.sim as the first simulation file and keeps the deterministic/sample-period/only-body chain.",
        ],
    }
    summary_path = output_dir / "benchmark_plan.summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {summary_path}. Use --overwrite if intended.")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared mini CPU benchmark: {output_dir}")
    print(f"Source run: {source_run_dir}")
    print(f"Seed rows: {len(manifest_rows)}")
    print(f"CPU jobs: {len(cpu_rows)}")
    print(f"Framsticks sim: {MINI_SIM_CHAIN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
