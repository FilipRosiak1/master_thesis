from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import difflib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1vae.config.defaults import DEFAULTS
from f1vae.config.io import maybe_resolve_path
from f1vae.data.datasets import GrammarRuleDataset
from f1vae.inference.decode import decode_grammar_indices
from f1vae.models.registry import build_model, dataset_class_for_model
from f1vae.training.loops import _beta_for_epoch, _format_duration, _tf_ratio_for_epoch
from f1vae.training.losses import sequence_vae_loss


TREE_MODELS = {
    "tree_vae",
    "tree_vae_masked",
    "tree_vae_masked_lhs",
    "tree_vae_masked_lhs_depth",
    "tree_vae_masked_lhs_cond",
}


class FitnessIndexedDataset(Dataset):
    def __init__(
        self,
        dataset: GrammarRuleDataset,
        indices: list[int],
        *,
        fitness_mean: float,
        fitness_std: float,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.fitness_mean = fitness_mean
        self.fitness_std = fitness_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_idx = self.indices[idx]
        raw = float(self.dataset.valid_fitnesses[source_idx])
        normalized = (raw - self.fitness_mean) / self.fitness_std
        return (
            self.dataset[source_idx],
            torch.tensor(normalized, dtype=torch.float32),
            torch.tensor(raw, dtype=torch.float32),
            torch.tensor(source_idx, dtype=torch.long),
        )


class FitnessHead(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fitness-aware or fitness-conditional F1 VAEs")
    parser.add_argument("--mode", choices=("auxiliary", "conditional"), required=True)
    parser.add_argument("--model", default=None, help="Base model for auxiliary, conditional model for conditional mode")
    parser.add_argument("--data-path", default="datasets/f1/f1_dataset.txt")
    parser.add_argument("--output-root", default="models/f1_guided")
    parser.add_argument("--run-dir", default=None, help="Exact output run directory")
    parser.add_argument("--run-name", default=None, help="Run name under output root")
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--schedule-epochs", type=int, default=None)
    parser.add_argument("--fitness-weight", type=float, default=100.0)
    parser.add_argument("--fitness-head-hidden", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-items", type=int, default=None, help="Optional smoke-test cap before split")
    return parser


def _finite_indices(dataset: GrammarRuleDataset) -> list[int]:
    indices: list[int] = []
    for idx, value in enumerate(dataset.valid_fitnesses):
        try:
            fitness = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isnan(fitness) and not math.isinf(fitness):
            indices.append(idx)
    return indices


def _split_indices(indices: list[int], val_split: float, seed: int | None) -> tuple[list[int], list[int]]:
    if not indices:
        raise RuntimeError("No finite fitness labels available")
    generator = torch.Generator().manual_seed(seed or 0)
    order = torch.randperm(len(indices), generator=generator).tolist()
    shuffled = [indices[i] for i in order]
    if val_split <= 0.0 or len(shuffled) < 2:
        return shuffled, []
    val_size = max(1, int(len(shuffled) * val_split))
    val_size = min(len(shuffled) - 1, val_size)
    return shuffled[val_size:], shuffled[:val_size]


def _make_run_dir(output_root: str, model_name: str, run_name: str | None, run_dir: str | None) -> Path:
    if run_dir is not None:
        path = Path(maybe_resolve_path(run_dir, root_dir=str(ROOT)))
    elif run_name is not None:
        path = Path(maybe_resolve_path(output_root, root_dir=str(ROOT))) / run_name
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = Path(maybe_resolve_path(output_root, root_dir=str(ROOT))) / model_name / stamp
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)
    return path


def _decode_batch(model_name: str, model, batch: torch.Tensor, fitness_norm: torch.Tensor | None) -> list[str]:
    if model_name not in TREE_MODELS:
        raise ValueError(f"Only tree/grammar-rule models are supported, got {model_name}")
    mu, _ = model.encode(batch)
    if hasattr(model, "condition_dim"):
        logits = model.decode(mu, None, teacher_forcing_ratio=0.0, fitness_condition=fitness_norm)
    else:
        logits = model.decode(mu, None, teacher_forcing_ratio=0.0)
    tokens = logits.argmax(dim=-1)
    return [decode_grammar_indices(tokens[i]) for i in range(tokens.size(0))]


def _reference_batch(batch: torch.Tensor) -> list[str]:
    return [decode_grammar_indices(batch[i]) for i in range(batch.size(0))]


def _evaluate_reconstruction(
    model_name: str,
    model,
    loader: DataLoader,
    device: torch.device,
    conditional: bool,
) -> tuple[int, int, float]:
    exact = 0
    total = 0
    similarity_sum = 0.0
    model.eval()
    with torch.no_grad():
        for batch, fitness_norm, _, _ in loader:
            batch = batch.to(device)
            fitness_norm = fitness_norm.to(device) if conditional else None
            originals = _reference_batch(batch)
            reconstructions = _decode_batch(model_name, model, batch, fitness_norm)
            for original, reconstructed in zip(originals, reconstructions):
                if original == reconstructed:
                    exact += 1
                similarity_sum += difflib.SequenceMatcher(None, original, reconstructed).ratio()
                total += 1
    return exact, total, similarity_sum / max(1, total)


def _evaluate_fitness_head(
    model,
    head: FitnessHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    predictions: list[float] = []
    targets: list[float] = []
    total_mse = 0.0
    total = 0
    model.eval()
    head.eval()
    with torch.no_grad():
        for batch, fitness_norm, _, _ in loader:
            batch = batch.to(device)
            fitness_norm = fitness_norm.to(device)
            mu, _ = model.encode(batch)
            pred = head(mu)
            total_mse += F.mse_loss(pred, fitness_norm, reduction="sum").item()
            total += batch.size(0)
            predictions.extend(pred.detach().cpu().tolist())
            targets.extend(fitness_norm.detach().cpu().tolist())
    corr = 0.0
    if len(predictions) > 1 and np.std(predictions) > 1e-8 and np.std(targets) > 1e-8:
        corr = float(np.corrcoef(np.asarray(predictions), np.asarray(targets))[0, 1])
    return total_mse / max(1, total), corr


def _log(run_dir: Path, line: str) -> None:
    print(line, flush=True)
    with (run_dir / "training.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    args = _build_parser().parse_args()
    model_name = args.model
    if model_name is None:
        model_name = "tree_vae_masked_lhs" if args.mode == "auxiliary" else "tree_vae_masked_lhs_cond"
    if args.mode == "conditional" and not model_name.endswith("_cond"):
        raise ValueError("Conditional mode requires a conditional model, e.g. tree_vae_masked_lhs_cond")
    if args.mode == "auxiliary" and model_name.endswith("_cond"):
        raise ValueError("Auxiliary mode should use a non-conditional base model")

    defaults = DEFAULTS[model_name]
    latent_dim = defaults.latent_dim if args.latent_dim is None else args.latent_dim
    hidden_dim = defaults.hidden_dim if args.hidden_dim is None else args.hidden_dim
    embedding_dim = defaults.embedding_dim if args.embedding_dim is None else args.embedding_dim
    batch_size = defaults.batch_size if args.batch_size is None else args.batch_size
    epochs = defaults.epochs if args.epochs is None else args.epochs
    learning_rate = defaults.learning_rate if args.learning_rate is None else args.learning_rate
    max_length = defaults.max_length if args.max_length is None else args.max_length
    schedule_epochs = epochs if args.schedule_epochs is None else args.schedule_epochs

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    data_path = maybe_resolve_path(args.data_path, root_dir=str(ROOT))
    dataset_cls = dataset_class_for_model(model_name)
    dataset = dataset_cls(data_path, max_length)
    if not isinstance(dataset, GrammarRuleDataset):
        raise TypeError(f"Only GrammarRuleDataset models are supported, got {type(dataset).__name__}")
    indices = _finite_indices(dataset)
    if args.max_items is not None:
        indices = indices[: args.max_items]
    train_indices, val_indices = _split_indices(indices, args.val_split, args.seed)
    train_fitness = np.asarray([float(dataset.valid_fitnesses[idx]) for idx in train_indices], dtype=np.float64)
    fitness_mean = float(train_fitness.mean())
    fitness_std = float(max(train_fitness.std(), 1e-8))

    run_dir = _make_run_dir(args.output_root, model_name, args.run_name, args.run_dir)
    config = {
        "mode": args.mode,
        "model": model_name,
        "data_path": data_path,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "checkpoint_every": args.checkpoint_every,
        "seed": args.seed,
        "val_split": args.val_split,
        "val_every": args.val_every,
        "schedule_epochs": schedule_epochs,
        "fitness_weight": args.fitness_weight,
        "fitness_head_hidden": args.fitness_head_hidden,
        "fitness_mean": fitness_mean,
        "fitness_std": fitness_std,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
    }
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(run_dir, f"Using device: {device}")
    _log(run_dir, f"Fitness normalization: mean={fitness_mean:.8g} std={fitness_std:.8g}")

    train_dataset = FitnessIndexedDataset(dataset, train_indices, fitness_mean=fitness_mean, fitness_std=fitness_std)
    val_dataset = FitnessIndexedDataset(dataset, val_indices, fitness_mean=fitness_mean, fitness_std=fitness_std) if val_indices else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_recon_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None

    model = build_model(
        model_name=model_name,
        dataset=dataset,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        max_length=max_length,
    ).to(device)
    fitness_head = None
    parameters: list[torch.nn.Parameter] = list(model.parameters())
    if args.mode == "auxiliary":
        fitness_head = FitnessHead(latent_dim, args.fitness_head_hidden).to(device)
        parameters.extend(fitness_head.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)

    checkpoint_meta = {
        "format_version": 2,
        "model_name": model_name,
        "data_path": data_path,
        "max_length": max_length,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "num_productions": len(dataset.productions),
        "fitness_guided_mode": args.mode,
        "fitness_mean": fitness_mean,
        "fitness_std": fitness_std,
    }

    best_val_exact = -1
    best_val_similarity = -1.0
    best_val_fitness_mse = float("inf")
    train_start = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        if fitness_head is not None:
            fitness_head.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kld = 0.0
        total_fit = 0.0
        seen = 0
        beta = _beta_for_epoch(epoch, schedule_epochs)
        tf_ratio = _tf_ratio_for_epoch(epoch, schedule_epochs)

        for batch, fitness_norm, _, _ in train_loader:
            batch = batch.to(device)
            fitness_norm = fitness_norm.to(device)
            optimizer.zero_grad()
            if args.mode == "conditional":
                recon_logits, mu, logvar = model(batch, fitness_norm, teacher_forcing_ratio=tf_ratio)
            else:
                recon_logits, mu, logvar = model(batch, teacher_forcing_ratio=tf_ratio)
            loss, ce, kld = sequence_vae_loss(recon_logits, batch, mu, logvar, pad_idx=dataset.pad_rule_idx, beta=beta)
            fitness_loss = torch.tensor(0.0, device=device)
            if fitness_head is not None:
                pred = fitness_head(mu)
                fitness_loss = F.mse_loss(pred, fitness_norm, reduction="sum")
                loss = loss + args.fitness_weight * fitness_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()

            batch_size_actual = batch.size(0)
            seen += batch_size_actual
            total_loss += loss.item()
            total_ce += ce.item()
            total_kld += kld.item()
            total_fit += fitness_loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _log(
                run_dir,
                f"Epoch {epoch + 1}/{epochs}\tLoss: {total_loss / max(1, seen):.4f}"
                f"\tCE: {total_ce / max(1, seen):.4f}\tKLD: {total_kld / max(1, seen):.4f}"
                f"\tFitMSE: {total_fit / max(1, seen):.4f}\tBeta: {beta:.2f}\tTF: {tf_ratio:.2f}",
            )

        if (epoch + 1) % args.checkpoint_every == 0:
            payload: dict[str, Any] = {"state_dict": model.state_dict(), "meta": checkpoint_meta}
            if fitness_head is not None:
                payload["fitness_head_state_dict"] = fitness_head.state_dict()
            torch.save(payload, run_dir / "checkpoints" / f"epoch_{epoch + 1}.pth")

        if val_loader is not None and ((epoch + 1) % args.val_every == 0 or epoch == 0 or (epoch + 1) == epochs):
            conditional = args.mode == "conditional"
            val_exact, val_total, val_similarity = _evaluate_reconstruction(model_name, model, val_loader, device, conditional)
            train_exact, train_total, train_similarity = _evaluate_reconstruction(
                model_name,
                model,
                train_recon_loader,
                device,
                conditional,
            )
            fit_mse = 0.0
            fit_corr = 0.0
            if fitness_head is not None:
                fit_mse, fit_corr = _evaluate_fitness_head(model, fitness_head, val_loader, device)

            _log(
                run_dir,
                f"TrainRecon {epoch + 1}/{epochs}\tExact: {train_exact}/{train_total}"
                f"\tExactAcc: {(100.0 * train_exact / max(1, train_total)):.2f}%"
                f"\tAvgSim: {(100.0 * train_similarity):.2f}%",
            )
            _log(
                run_dir,
                f"ValRecon {epoch + 1}/{epochs}\tExact: {val_exact}/{val_total}"
                f"\tExactAcc: {(100.0 * val_exact / max(1, val_total)):.2f}%"
                f"\tAvgSim: {(100.0 * val_similarity):.2f}%\tFitMSE: {fit_mse:.4f}\tFitCorr: {fit_corr:.4f}",
            )

            if val_exact > best_val_exact or (val_exact == best_val_exact and val_similarity > best_val_similarity):
                best_val_exact = val_exact
                best_val_similarity = val_similarity
                best_meta = {
                    **checkpoint_meta,
                    "best_epoch": epoch + 1,
                    "train_exact": train_exact,
                    "train_total": train_total,
                    "train_exact_acc": train_exact / max(1, train_total),
                    "train_avg_similarity": train_similarity,
                    "val_exact": val_exact,
                    "val_total": val_total,
                    "val_exact_acc": val_exact / max(1, val_total),
                    "val_avg_similarity": val_similarity,
                    "val_fitness_mse": fit_mse,
                    "val_fitness_corr": fit_corr,
                }
                payload = {"state_dict": model.state_dict(), "meta": best_meta}
                if fitness_head is not None:
                    payload["fitness_head_state_dict"] = fitness_head.state_dict()
                torch.save(payload, run_dir / "best_val_recon.pth")
                _log(run_dir, f"BestValRecon {epoch + 1}/{epochs}\tExact: {val_exact}/{val_total}\tAvgSim: {(100.0 * val_similarity):.2f}%")

            if fitness_head is not None and fit_mse < best_val_fitness_mse:
                best_val_fitness_mse = fit_mse
                payload = {"state_dict": model.state_dict(), "fitness_head_state_dict": fitness_head.state_dict(), "meta": checkpoint_meta}
                torch.save(payload, run_dir / "best_val_fitness.pth")
                _log(run_dir, f"BestValFitness {epoch + 1}/{epochs}\tMSE: {fit_mse:.4f}\tCorr: {fit_corr:.4f}")

        elapsed = time.perf_counter() - train_start
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_epoch_time = elapsed / (epoch + 1)
            remaining = avg_epoch_time * (epochs - (epoch + 1))
            _log(
                run_dir,
                f"Progress {epoch + 1}/{epochs}\tEpoch time: {_format_duration(time.perf_counter() - epoch_start)}"
                f"\tElapsed: {_format_duration(elapsed)}\tETA: {_format_duration(remaining)}",
            )

    final_payload: dict[str, Any] = {"state_dict": model.state_dict(), "meta": checkpoint_meta}
    if fitness_head is not None:
        final_payload["fitness_head_state_dict"] = fitness_head.state_dict()
        torch.save(fitness_head.state_dict(), run_dir / "fitness_head.pth")
    final_path = run_dir / f"{model_name}.pth"
    torch.save(final_payload, final_path)
    _log(run_dir, f"Saved model to: {final_path}")


if __name__ == "__main__":
    main()
