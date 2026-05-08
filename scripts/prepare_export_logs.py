from __future__ import annotations

import argparse
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class LogRun:
    label: str
    run_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare export package with newest training logs from selected model/run directories"
    )
    parser.add_argument(
        "--models-root",
        default="models/f1_val",
        help="Root directory containing model subdirectories or nested sweep directories",
    )
    parser.add_argument(
        "--exports-dir",
        default="exports",
        help="Directory where export folder and archive will be created",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="Comma-separated indices of log runs to export (example: 1,3,4). If omitted, interactive prompt is shown.",
    )
    return parser.parse_args()


def _list_latest_log_runs(models_root: Path) -> list[LogRun]:
    if not models_root.exists() or not models_root.is_dir():
        raise FileNotFoundError(f"Models root not found: {models_root}")

    latest_by_label: dict[str, Path] = {}
    for training_log in models_root.rglob("training.log"):
        if not training_log.is_file():
            continue
        run_dir = training_log.parent
        label = run_dir.parent.relative_to(models_root).as_posix()
        previous = latest_by_label.get(label)
        if previous is None or run_dir.stat().st_mtime > previous.stat().st_mtime:
            latest_by_label[label] = run_dir

    return [LogRun(label, latest_by_label[label]) for label in sorted(latest_by_label)]


def _parse_selection(raw: str, max_idx: int) -> list[int]:
    indices: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        idx = int(text)
        if idx < 1 or idx > max_idx:
            raise ValueError(f"Index out of range: {idx}")
        if idx not in indices:
            indices.append(idx)
    if not indices:
        raise ValueError("No valid indices selected")
    return indices


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    models_root = (repo_root / args.models_root).resolve()
    exports_dir = (repo_root / args.exports_dir).resolve()
    exports_dir.mkdir(parents=True, exist_ok=True)

    log_runs = _list_latest_log_runs(models_root)
    if not log_runs:
        raise RuntimeError(f"No training logs found in {models_root}")

    print("Available latest training logs:")
    for idx, log_run in enumerate(log_runs, start=1):
        print(f"  {idx}. {log_run.label} (newest: {log_run.run_dir.name})")

    if args.select is None:
        raw = input("Select log indices to export (example: 1,3,4): ").strip()
    else:
        raw = args.select.strip()

    selected_indices = _parse_selection(raw, len(log_runs))
    selected_runs = [log_runs[i - 1] for i in selected_indices]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = exports_dir / f"f1_val_logs_{stamp}"
    export_folder.mkdir(parents=True, exist_ok=False)

    copied = 0
    skipped: list[str] = []

    for log_run in selected_runs:
        training_log = log_run.run_dir / "training.log"
        run_config = log_run.run_dir / "run_config.json"
        if not training_log.exists():
            skipped.append(f"{log_run.label}: missing training.log in {log_run.run_dir.name}")
            continue

        target_dir = export_folder / log_run.label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(training_log, target_dir / "training.log")
        if run_config.exists():
            shutil.copy2(run_config, target_dir / "run_config.json")

        source_file = target_dir / "source_run.txt"
        source_file.write_text(str(log_run.run_dir) + "\n", encoding="utf-8")
        copied += 1

    archive_path = export_folder.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(export_folder, arcname=export_folder.name)

    print("\nExport ready")
    print(f"- Folder : {export_folder}")
    print(f"- Archive: {archive_path}")
    print(f"- Models exported: {copied}")
    if skipped:
        print("- Skipped:")
        for item in skipped:
            print(f"  * {item}")


if __name__ == "__main__":
    main()
