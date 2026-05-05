from __future__ import annotations

import argparse
import shutil
import tarfile
from datetime import datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare export package with selected checkpoints from latest runs of chosen models"
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
        "--models",
        default=None,
        help="Comma-separated model indices (example: 1,3,4). If omitted, interactive prompt is shown.",
    )
    return parser.parse_args()


def _list_model_dirs(models_root: Path) -> list[Path]:
    if not models_root.exists() or not models_root.is_dir():
        raise FileNotFoundError(f"Models root not found: {models_root}")
    return sorted([p for p in models_root.iterdir() if p.is_dir()], key=lambda p: p.name)


def _latest_run_dir(model_dir: Path) -> Path | None:
    runs = [p for p in model_dir.iterdir() if p.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _checkpoint_files(run_dir: Path) -> list[Path]:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists() or not ckpt_dir.is_dir():
        return []
    return sorted(
        [p for p in ckpt_dir.glob("epoch_*.pth") if p.is_file()],
        key=lambda p: p.name,
    )


def _parse_indices(raw: str, max_idx: int) -> list[int]:
    out: list[int] = []
    for chunk in raw.split(","):
        txt = chunk.strip()
        if not txt:
            continue
        idx = int(txt)
        if idx < 1 or idx > max_idx:
            raise ValueError(f"Index out of range: {idx}")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("No valid indices selected")
    return out


def _pick_checkpoints_interactive(model_name: str, ckpts: list[Path]) -> list[Path]:
    print(f"\nModel: {model_name}")
    print("Available checkpoints from latest run:")
    for idx, ckpt in enumerate(ckpts, start=1):
        print(f"  {idx}. {ckpt.name}")
    raw = input("Select checkpoint indices (example: 1,3,5 or 'all'): ").strip().lower()
    if raw == "all":
        return ckpts
    chosen_idx = _parse_indices(raw, len(ckpts))
    return [ckpts[i - 1] for i in chosen_idx]


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
        latest = _latest_run_dir(model_dir)
        latest_name = latest.name if latest is not None else "<no runs>"
        print(f"  {idx}. {model_dir.name} (latest: {latest_name})")

    if args.models is None:
        raw_models = input("Select model indices (example: 1,3,4): ").strip()
    else:
        raw_models = args.models.strip()
    model_idx = _parse_indices(raw_models, len(model_dirs))
    selected_models = [model_dirs[i - 1] for i in model_idx]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = exports_dir / f"f1_val_ckpts_{stamp}"
    export_folder.mkdir(parents=True, exist_ok=False)

    manifest_lines: list[str] = []
    copied = 0
    skipped: list[str] = []

    for model_dir in selected_models:
        latest = _latest_run_dir(model_dir)
        if latest is None:
            skipped.append(f"{model_dir.name}: no runs")
            continue

        ckpts = _checkpoint_files(latest)
        if not ckpts:
            skipped.append(f"{model_dir.name}: no checkpoints in latest run {latest.name}")
            continue

        selected_ckpts = _pick_checkpoints_interactive(model_dir.name, ckpts)
        if not selected_ckpts:
            skipped.append(f"{model_dir.name}: no checkpoints selected")
            continue

        target_run_dir = export_folder / model_dir.name / latest.name
        target_ckpt_dir = target_run_dir / "checkpoints"
        target_ckpt_dir.mkdir(parents=True, exist_ok=True)

        for ckpt in selected_ckpts:
            shutil.copy2(ckpt, target_ckpt_dir / ckpt.name)
            manifest_lines.append(f"{model_dir.name}/{latest.name}/{ckpt.name} <- {ckpt}")
            copied += 1

        run_config = latest / "run_config.json"
        training_log = latest / "training.log"
        if run_config.exists():
            shutil.copy2(run_config, target_run_dir / "run_config.json")
        if training_log.exists():
            shutil.copy2(training_log, target_run_dir / "training.log")
        (target_run_dir / "source_run.txt").write_text(str(latest) + "\n", encoding="utf-8")

    (export_folder / "manifest.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    archive_path = export_folder.with_suffix(".tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(export_folder, arcname=export_folder.name)

    print("\nExport ready")
    print(f"- Folder : {export_folder}")
    print(f"- Archive: {archive_path}")
    print(f"- Checkpoints exported: {copied}")
    if skipped:
        print("- Skipped:")
        for item in skipped:
            print(f"  * {item}")


if __name__ == "__main__":
    main()
