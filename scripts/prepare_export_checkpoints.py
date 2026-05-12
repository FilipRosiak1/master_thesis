from __future__ import annotations

import argparse
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class CheckpointRun:
    label: str
    run_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare export package with selected checkpoints from newest model runs"
    )
    parser.add_argument(
        "--models-root",
        default="models/f1_val,models/f1_sweep,models/f1_sweep_lhs",
        help="Comma-separated root directories containing model subdirectories or nested sweep directories",
    )
    parser.add_argument(
        "--exports-dir",
        default="exports",
        help="Directory where export folder and archive will be created",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="Comma-separated run indices to export, or 'all'. If omitted, interactive prompt is shown.",
    )
    parser.add_argument(
        "--checkpoints",
        default="best_val_recon",
        help=(
            "Comma-separated checkpoint selectors: best_val_recon, final, latest_epoch, all, "
            "or exact .pth file names. Default: best_val_recon."
        ),
    )
    return parser.parse_args()


def _root_label(root: Path, repo_root: Path) -> str:
    models_dir = repo_root / "models"
    try:
        return root.relative_to(models_dir).as_posix()
    except ValueError:
        try:
            return root.relative_to(repo_root).as_posix()
        except ValueError:
            return root.name


def _resolve_model_roots(raw: str, repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        root = (repo_root / text).resolve()
        if root.exists() and root.is_dir() and root not in roots:
            roots.append(root)
    if not roots:
        raise FileNotFoundError(f"No model roots found from: {raw}")
    return roots


def _epoch_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("epoch_"):
        suffix = stem.removeprefix("epoch_")
        if suffix.isdigit():
            return int(suffix), path.name
    return -1, path.name


def _all_checkpoint_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    files.extend(sorted([p for p in run_dir.glob("*.pth") if p.is_file()], key=lambda p: p.name))

    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists() and ckpt_dir.is_dir():
        files.extend(sorted([p for p in ckpt_dir.glob("*.pth") if p.is_file()], key=_epoch_key))

    return files


def _list_latest_checkpoint_runs(models_roots: list[Path], repo_root: Path) -> list[CheckpointRun]:
    include_root_label = len(models_roots) > 1
    all_runs: list[CheckpointRun] = []

    for models_root in models_roots:
        all_runs.extend(
            _list_latest_checkpoint_runs_for_root(models_root, repo_root, include_root_label=include_root_label)
        )

    return sorted(all_runs, key=lambda run: run.run_dir.stat().st_mtime, reverse=True)


def _list_latest_checkpoint_runs_for_root(
    models_root: Path,
    repo_root: Path,
    *,
    include_root_label: bool,
) -> list[CheckpointRun]:
    if not models_root.exists() or not models_root.is_dir():
        raise FileNotFoundError(f"Models root not found: {models_root}")

    latest_by_label: dict[str, Path] = {}
    for run_config in models_root.rglob("run_config.json"):
        if not run_config.is_file():
            continue
        run_dir = run_config.parent
        if not _all_checkpoint_files(run_dir):
            continue

        label = run_dir.parent.relative_to(models_root).as_posix()
        if include_root_label:
            label = f"{_root_label(models_root, repo_root)}/{label}"

        previous = latest_by_label.get(label)
        if previous is None or run_dir.stat().st_mtime > previous.stat().st_mtime:
            latest_by_label[label] = run_dir

    return [CheckpointRun(label, latest_by_label[label]) for label in sorted(latest_by_label)]


def _parse_selection(raw: str, max_idx: int) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(1, max_idx + 1))

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


def _checkpoint_files_for_selector(run_dir: Path, selector: str) -> list[Path]:
    if selector == "all":
        return _all_checkpoint_files(run_dir)

    if selector == "best_val_recon":
        path = run_dir / "best_val_recon.pth"
        return [path] if path.exists() and path.is_file() else []

    if selector == "final":
        return sorted(
            [p for p in run_dir.glob("*.pth") if p.is_file() and p.name != "best_val_recon.pth"],
            key=lambda p: p.name,
        )

    if selector == "latest_epoch":
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            return []
        epoch_files = sorted([p for p in ckpt_dir.glob("epoch_*.pth") if p.is_file()], key=_epoch_key)
        return epoch_files[-1:] if epoch_files else []

    direct_path = run_dir / selector
    if direct_path.exists() and direct_path.is_file():
        return [direct_path]

    checkpoint_path = run_dir / "checkpoints" / selector
    if checkpoint_path.exists() and checkpoint_path.is_file():
        return [checkpoint_path]

    return []


def _selected_checkpoint_files(run_dir: Path, selectors: list[str]) -> list[Path]:
    selected: list[Path] = []
    for selector in selectors:
        for path in _checkpoint_files_for_selector(run_dir, selector):
            if path not in selected:
                selected.append(path)
    return selected


def _parse_checkpoint_selectors(raw: str) -> list[str]:
    selectors = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not selectors:
        raise ValueError("No checkpoint selectors provided")
    return selectors


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    models_roots = _resolve_model_roots(args.models_root, repo_root)
    exports_dir = (repo_root / args.exports_dir).resolve()
    exports_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_runs = _list_latest_checkpoint_runs(models_roots, repo_root)
    if not checkpoint_runs:
        roots_text = ", ".join(str(root) for root in models_roots)
        raise RuntimeError(f"No checkpointed runs found in: {roots_text}")

    print("Available latest checkpointed runs:")
    for idx, checkpoint_run in enumerate(checkpoint_runs, start=1):
        print(f"  {idx}. {checkpoint_run.label} (newest: {checkpoint_run.run_dir.name})")

    if args.select is None:
        raw = input("Select run indices to export (example: 1,3,4 or all): ").strip()
    else:
        raw = args.select.strip()

    selected_indices = _parse_selection(raw, len(checkpoint_runs))
    selected_runs = [checkpoint_runs[i - 1] for i in selected_indices]
    checkpoint_selectors = _parse_checkpoint_selectors(args.checkpoints)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_folder = exports_dir / f"f1_val_ckpts_{stamp}"
    export_folder.mkdir(parents=True, exist_ok=False)

    manifest_lines: list[str] = []
    copied = 0
    skipped: list[str] = []

    for checkpoint_run in selected_runs:
        selected_ckpts = _selected_checkpoint_files(checkpoint_run.run_dir, checkpoint_selectors)
        if not selected_ckpts:
            skipped.append(
                f"{checkpoint_run.label}: no checkpoints matching {', '.join(checkpoint_selectors)} "
                f"in {checkpoint_run.run_dir.name}"
            )
            continue

        target_run_dir = export_folder / checkpoint_run.label
        target_run_dir.mkdir(parents=True, exist_ok=True)

        for ckpt in selected_ckpts:
            relative_path = ckpt.relative_to(checkpoint_run.run_dir)
            target_path = target_run_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt, target_path)
            manifest_lines.append(f"{checkpoint_run.label}/{relative_path.as_posix()} <- {ckpt}")
            copied += 1

        run_config = checkpoint_run.run_dir / "run_config.json"
        training_log = checkpoint_run.run_dir / "training.log"
        if run_config.exists():
            shutil.copy2(run_config, target_run_dir / "run_config.json")
        if training_log.exists():
            shutil.copy2(training_log, target_run_dir / "training.log")
        (target_run_dir / "source_run.txt").write_text(str(checkpoint_run.run_dir) + "\n", encoding="utf-8")

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
