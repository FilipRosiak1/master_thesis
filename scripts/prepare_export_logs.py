from __future__ import annotations

import argparse
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare export package with newest training logs from selected model directories"
    )
    parser.add_argument(
        "--models-root",
        default="models/f1_val",
        help="Root directory containing model subdirectories",
    )
    parser.add_argument(
        "--exports-dir",
        default="exports",
        help="Directory where export folder and archive will be created",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="Comma-separated indices of models to export (example: 1,3,4). If omitted, interactive prompt is shown.",
    )
    return parser.parse_args()


def _list_model_dirs(models_root: Path) -> list[Path]:
    if not models_root.exists() or not models_root.is_dir():
        raise FileNotFoundError(f"Models root not found: {models_root}")
    return sorted([p for p in models_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def _newest_run_dir(model_dir: Path) -> Path | None:
    run_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


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

    model_dirs = _list_model_dirs(models_root)
    if not model_dirs:
        raise RuntimeError(f"No model directories found in {models_root}")

    print("Available model directories:")
    for idx, model_dir in enumerate(model_dirs, start=1):
        newest = _newest_run_dir(model_dir)
        newest_name = newest.name if newest is not None else "<no runs>"
        print(f"  {idx}. {model_dir.name} (newest: {newest_name})")

    if args.select is None:
        raw = input("Select model indices to export (example: 1,3,4): ").strip()
    else:
        raw = args.select.strip()

    selected_indices = _parse_selection(raw, len(model_dirs))
    selected_models = [model_dirs[i - 1] for i in selected_indices]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = exports_dir / f"f1_val_logs_{stamp}"
    export_folder.mkdir(parents=True, exist_ok=False)

    copied = 0
    skipped: list[str] = []

    for model_dir in selected_models:
        newest = _newest_run_dir(model_dir)
        if newest is None:
            skipped.append(f"{model_dir.name}: no run directories")
            continue

        training_log = newest / "training.log"
        run_config = newest / "run_config.json"
        if not training_log.exists():
            skipped.append(f"{model_dir.name}: missing training.log in {newest.name}")
            continue

        target_dir = export_folder / model_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(training_log, target_dir / "training.log")
        if run_config.exists():
            shutil.copy2(run_config, target_dir / "run_config.json")

        source_file = target_dir / "source_run.txt"
        source_file.write_text(str(newest) + "\n", encoding="utf-8")
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
