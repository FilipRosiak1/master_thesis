from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1vae.utils import command_run_logger

SCRIPT_MAP = {
    "train": "train.py",
    "eval": "eval.py",
    "infer": "infer.py",
    "optimize": "optimize_latent.py",
}


def _run_python_script(script_name: str, args: list[str]) -> int:
    script_path = SCRIPTS / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    cmd = [sys.executable, str(script_path), *args]
    print(f"\n$ {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=str(ROOT), check=False)
    return completed.returncode


def _latest_weights_path(output_root: str, model_name: str) -> str:
    model_dir = ROOT / output_root / model_name
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Could not find model directory '{model_dir}'. Train step may have failed."
        )

    run_dirs = [p for p in model_dir.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in '{model_dir}'.")

    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    weights = latest_run / f"{model_name}.pth"
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    return str(weights)


def _pipeline(args: argparse.Namespace) -> int:
    train_args = [
        "--model",
        args.model,
        "--data-path",
        args.data_path,
        "--output-root",
        args.output_root,
    ]

    if args.config_model:
        train_args.extend(["--config-model", args.config_model])
    if args.config_data:
        train_args.extend(["--config-data", args.config_data])
    if args.config_train:
        train_args.extend(["--config-train", args.config_train])
    if args.epochs is not None:
        train_args.extend(["--epochs", str(args.epochs)])
    if args.seed is not None:
        train_args.extend(["--seed", str(args.seed)])

    rc = _run_python_script("train.py", train_args)
    if rc != 0:
        return rc

    weights = _latest_weights_path(args.output_root, args.model)
    print(f"\nDetected latest weights: {weights}")

    eval_reconstruct = [
        "--model",
        args.model,
        "--weights",
        weights,
        "--data-path",
        args.data_path,
        "--mode",
        "reconstruct",
    ]
    rc = _run_python_script("eval.py", eval_reconstruct)
    if rc != 0:
        return rc

    eval_mutate = [
        "--model",
        args.model,
        "--weights",
        weights,
        "--data-path",
        args.data_path,
        "--mode",
        "mutate",
        "--num-samples",
        str(args.mutation_samples),
        "--noise-scale",
        str(args.noise_scale),
    ]
    rc = _run_python_script("eval.py", eval_mutate)
    if rc != 0:
        return rc

    if args.fitness_fn:
        optimize_args = [
            "--model",
            args.model,
            "--weights",
            weights,
            "--fitness-fn",
            args.fitness_fn,
            "--algorithm",
            args.algorithm,
            "--iterations",
            str(args.optimize_iterations),
        ]
        if args.population_size is not None:
            optimize_args.extend(["--population-size", str(args.population_size)])
        rc = _run_python_script("optimize_latent.py", optimize_args)
        if rc != 0:
            return rc

    print("\nPipeline finished successfully.")
    return 0


def _interactive_menu() -> int:
    options = [
        ("train", "Run training"),
        ("eval", "Run evaluation"),
        ("infer", "Run single-sample inference"),
        ("optimize", "Run latent optimization"),
        ("pipeline", "Train + evaluate (+ optimize if configured)"),
    ]
    print("F1 VAE Workbench")
    for idx, (_, label) in enumerate(options, start=1):
        print(f"{idx}. {label}")
    print("0. Exit")

    raw = input("Select action: ").strip()
    if raw == "0":
        return 0

    try:
        choice = int(raw)
    except ValueError:
        print("Invalid selection")
        return 1

    if choice < 1 or choice > len(options):
        print("Invalid selection")
        return 1

    command = options[choice - 1][0]
    if command == "pipeline":
        model = input("Model [char_vae/grammar_vae/grammar_vae_masked/tree_vae/transformer_vae/vq_grammar_ae]: ").strip()
        data_path = input("Data path [datasets/f1/f1_dataset.txt]: ").strip() or "datasets/f1/f1_dataset.txt"
        namespace = argparse.Namespace(
            model=model,
            data_path=data_path,
            output_root="models/f1",
            config_model=None,
            config_data=None,
            config_train=None,
            epochs=None,
            seed=None,
            mutation_samples=10,
            noise_scale=1.0,
            fitness_fn=None,
            algorithm="cmaes",
            optimize_iterations=80,
            population_size=None,
        )
        return _pipeline(namespace)

    extra = input("Extra args (optional, write exactly like normal CLI): ").strip()
    forwarded = extra.split() if extra else []
    return _run_python_script(SCRIPT_MAP[command], forwarded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified workbench for train/eval/infer/optimize workflows")
    subparsers = parser.add_subparsers(dest="command")

    for command, script_name in SCRIPT_MAP.items():
        sub = subparsers.add_parser(command, help=f"Run scripts/{script_name}")
        sub.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the target script")

    pipe = subparsers.add_parser("pipeline", help="Run train -> eval reconstruct -> eval mutate -> optional optimize")
    pipe.add_argument("--model", required=True)
    pipe.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    pipe.add_argument("--output-root", default="models/f1")
    pipe.add_argument("--config-model", default=None)
    pipe.add_argument("--config-data", default=None)
    pipe.add_argument("--config-train", default=None)
    pipe.add_argument("--epochs", type=int, default=None)
    pipe.add_argument("--seed", type=int, default=None)
    pipe.add_argument("--mutation-samples", type=int, default=10)
    pipe.add_argument("--noise-scale", type=float, default=1.0)
    pipe.add_argument("--fitness-fn", default=None)
    pipe.add_argument("--algorithm", choices=("cmaes", "cem"), default="cmaes")
    pipe.add_argument("--optimize-iterations", type=int, default=80)
    pipe.add_argument("--population-size", type=int, default=None)

    subparsers.add_parser("menu", help="Interactive menu mode")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == "menu":
        raise SystemExit(_interactive_menu())

    if args.command == "pipeline":
        raise SystemExit(_pipeline(args))

    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    script_name = SCRIPT_MAP[args.command]
    raise SystemExit(_run_python_script(script_name, forwarded))


if __name__ == "__main__":
    with command_run_logger("scripts/workbench.py"):
        main()
