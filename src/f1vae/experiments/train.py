from __future__ import annotations

import argparse
from pathlib import Path

from f1vae.config.io import load_yaml, maybe_resolve_path
from f1vae.models.registry import MODEL_NAMES
from f1vae.training.loops import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train F1 VAE models from a shared codebase")
    parser.add_argument("--config-model", default=None, help="Path to model YAML config")
    parser.add_argument("--config-data", default=None, help="Path to data YAML config")
    parser.add_argument("--config-train", default=None, help="Path to train YAML config")
    parser.add_argument("--model", choices=MODEL_NAMES, required=False)
    parser.add_argument("--data-path", required=False)
    parser.add_argument("--output-root", default="models/f1")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--schedule-epochs", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_cfg = load_yaml(args.config_model)
    data_cfg = load_yaml(args.config_data)
    train_cfg = load_yaml(args.config_train)

    root_dir = str(Path(__file__).resolve().parents[3])

    model_name = args.model if args.model is not None else model_cfg.get("model")
    if model_name is None:
        raise ValueError("Model name is required. Provide --model or --config-model with a 'model' key.")

    data_path = args.data_path if args.data_path is not None else data_cfg.get("dataset_path")
    if data_path is None:
        raise ValueError("Data path is required. Provide --data-path or --config-data with a 'dataset_path' key.")

    data_path = maybe_resolve_path(data_path, root_dir=root_dir)

    output_root = args.output_root

    latent_dim = args.latent_dim if args.latent_dim is not None else model_cfg.get("latent_dim")
    hidden_dim = args.hidden_dim if args.hidden_dim is not None else model_cfg.get("hidden_dim")
    embedding_dim = args.embedding_dim if args.embedding_dim is not None else model_cfg.get("embedding_dim")
    max_length = args.max_length if args.max_length is not None else model_cfg.get("max_length")

    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size")
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs")
    learning_rate = args.learning_rate if args.learning_rate is not None else train_cfg.get("learning_rate")
    checkpoint_every = args.checkpoint_every if args.checkpoint_every is not None else train_cfg.get("checkpoint_every", 50)
    val_split = args.val_split if args.val_split is not None else train_cfg.get("val_split", 0.0)
    val_every = args.val_every if args.val_every is not None else train_cfg.get("val_every", 10)
    schedule_epochs = args.schedule_epochs if args.schedule_epochs is not None else train_cfg.get("schedule_epochs")

    train_model(
        model_name=model_name,
        data_path=data_path,
        output_root=output_root,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        max_length=max_length,
        checkpoint_every=checkpoint_every,
        seed=args.seed,
        val_split=val_split,
        val_every=val_every,
        schedule_epochs=schedule_epochs,
    )


if __name__ == "__main__":
    main()
