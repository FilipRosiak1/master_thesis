from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.config.defaults import DEFAULTS
from f1vae.config.io import load_yaml, maybe_resolve_path
from f1vae.inference.checkpoints import load_checkpoint
from f1vae.inference.evaluate import mutation_examples, reconstruction_metrics
from f1vae.models.registry import MODEL_NAMES, build_model, dataset_class_for_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate trained F1 VAE models")
    parser.add_argument("--config-model", default=None, help="Path to model YAML config")
    parser.add_argument("--config-data", default=None, help="Path to data YAML config")
    parser.add_argument("--model", choices=MODEL_NAMES, required=False)
    parser.add_argument("--weights", required=True, help="Path to .pth state_dict")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--mode", choices=("reconstruct", "mutate"), default="reconstruct")
    parser.add_argument("--device", default=None, help="cpu or cuda (defaults to auto)")
    parser.add_argument("--max-items", type=int, default=None, help="Optional reconstruction sample cap")
    parser.add_argument("--num-samples", type=int, default=10, help="Mutation sample count")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    model_cfg = load_yaml(args.config_model)
    data_cfg = load_yaml(args.config_data)
    root_dir = str(Path(__file__).resolve().parents[1])

    model_name = args.model if args.model is not None else model_cfg.get("model")
    if model_name is None:
        raise ValueError("Model name is required. Provide --model or --config-model with a 'model' key.")

    data_path = args.data_path if args.data_path is not None else data_cfg.get("dataset_path")
    if data_path is None:
        data_path = os.path.join(ROOT, "datasets", "f1", "f1_dataset.txt")
    data_path = maybe_resolve_path(data_path, root_dir=root_dir)

    defaults = DEFAULTS[model_name]
    latent_dim = model_cfg.get("latent_dim", defaults.latent_dim) if args.latent_dim is None else args.latent_dim
    hidden_dim = model_cfg.get("hidden_dim", defaults.hidden_dim) if args.hidden_dim is None else args.hidden_dim
    embedding_dim = model_cfg.get("embedding_dim", defaults.embedding_dim) if args.embedding_dim is None else args.embedding_dim
    max_length = model_cfg.get("max_length", defaults.max_length) if args.max_length is None else args.max_length

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cls = dataset_class_for_model(model_name)
    dataset = dataset_cls(data_path, max_length)

    model = build_model(
        model_name=model_name,
        dataset=dataset,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        max_length=max_length,
    ).to(device)

    state_dict, _ = load_checkpoint(args.weights, device)

    model.load_state_dict(state_dict)
    model.eval()

    if args.mode == "reconstruct":
        metrics = reconstruction_metrics(
            model_name=model_name,
            model=model,
            dataset=dataset,
            device=device,
            max_items=args.max_items,
        )
        print("Reconstruction results")
        print(f"  Total: {metrics['total']}")
        print(f"  Exact: {metrics['exact']}")
        print(f"  Exact accuracy: {metrics['exact_accuracy']:.2f}%")
        print(f"  Avg similarity: {metrics['avg_similarity']:.2f}%")
        return

    samples = mutation_examples(
        model_name=model_name,
        model=model,
        dataset=dataset,
        device=device,
        num_samples=args.num_samples,
        noise_scale=args.noise_scale,
    )
    print(f"Mutation samples ({len(samples)})")
    for i, (original, mutated) in enumerate(samples, start=1):
        print(f"{i:02d}. Original: {original}")
        print(f"    Mutated : {mutated}")


if __name__ == "__main__":
    main()
