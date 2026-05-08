from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.utils import command_run_logger

ADVANCED_MODELS = ["tree_vae", "tree_vae_masked", "tree_vae_masked_lhs", "transformer_vae", "vq_grammar_ae"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train advanced models with validation enabled"
    )
    parser.add_argument(
        "--data-path",
        default="datasets/f1/f1_dataset_1k_same.txt",
        help="Dataset path used for all runs",
    )
    parser.add_argument(
        "--output-root",
        default="models/f1",
        help="Root output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Epochs per model",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="Validation interval in epochs",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Optional checkpoint interval override",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed forwarded to each run",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue with next model if one run fails",
    )
    return parser


def _run_one(args: argparse.Namespace, model_name: str) -> int:
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "train.py"),
        "--model",
        model_name,
        "--data-path",
        args.data_path,
        "--output-root",
        args.output_root,
        "--epochs",
        str(args.epochs),
        "--val-split",
        str(args.val_split),
        "--val-every",
        str(args.val_every),
    ]
    if args.checkpoint_every is not None:
        cmd.extend(["--checkpoint-every", str(args.checkpoint_every)])
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])

    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"\n[{stamp}] START model={model_name}")
    print("$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    stamp_end = datetime.now().isoformat(timespec="seconds")
    print(f"[{stamp_end}] END   model={model_name} exit_code={completed.returncode}")
    return completed.returncode


def main() -> None:
    args = _build_parser().parse_args()

    failures: list[tuple[str, int]] = []
    for model_name in ADVANCED_MODELS:
        code = _run_one(args, model_name)
        if code != 0:
            failures.append((model_name, code))
            if not args.continue_on_failure:
                break

    print("\n=== Summary ===")
    if not failures:
        print("All advanced model trainings with validation finished successfully.")
        return

    print("Failures:")
    for model_name, code in failures:
        print(f"- {model_name}: exit_code={code}")
    raise SystemExit(1)


if __name__ == "__main__":
    with command_run_logger("scripts/train_advanced_models_with_validation.py"):
        main()
