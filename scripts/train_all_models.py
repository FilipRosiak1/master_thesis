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

from f1vae.models.registry import MODEL_NAMES


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially train all models with the same settings"
    )
    parser.add_argument(
        "--data-path",
        default="datasets/f1/f1_dataset.txt",
        help="Path to dataset used for every model",
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
        help="Number of epochs per model",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="Optional checkpoint frequency override",
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


def _run_one(
    *,
    model_name: str,
    data_path: str,
    output_root: str,
    epochs: int,
    checkpoint_every: int | None,
    seed: int | None,
) -> int:
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "train.py"),
        "--model",
        model_name,
        "--data-path",
        data_path,
        "--output-root",
        output_root,
        "--epochs",
        str(epochs),
    ]
    if checkpoint_every is not None:
        cmd.extend(["--checkpoint-every", str(checkpoint_every)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    stamp = datetime.now().isoformat(timespec="seconds")
    print(f"\n[{stamp}] START model={model_name}")
    print("$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=ROOT, check=False)
    stamp_end = datetime.now().isoformat(timespec="seconds")
    print(f"[{stamp_end}] END   model={model_name} exit_code={completed.returncode}")
    return completed.returncode


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    failures: list[tuple[str, int]] = []
    for model_name in MODEL_NAMES:
        code = _run_one(
            model_name=model_name,
            data_path=args.data_path,
            output_root=args.output_root,
            epochs=args.epochs,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed,
        )
        if code != 0:
            failures.append((model_name, code))
            if not args.continue_on_failure:
                break

    print("\n=== Summary ===")
    if not failures:
        print("All model trainings finished successfully.")
        return

    print("Failures:")
    for model_name, code in failures:
        print(f"- {model_name}: exit_code={code}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
