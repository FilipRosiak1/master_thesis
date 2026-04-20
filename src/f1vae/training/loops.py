from __future__ import annotations

import json
import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from f1vae.config.defaults import DEFAULTS
from f1vae.models.registry import build_model, dataset_class_for_model
from f1vae.training.losses import grammar_masked_vae_loss, sequence_vae_loss


def _make_run_dir(output_root: str, model_name: str) -> tuple[str, str]:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = os.path.join(output_root, model_name, stamp)
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    return base_dir, checkpoints_dir


def _beta_for_epoch(epoch: int, epochs: int) -> float:
    return min(1.0, epoch / (epochs * 0.5))


def _tf_ratio_for_epoch(epoch: int, epochs: int) -> float:
    return max(0.0, 1.0 - (epoch / (epochs * 0.75)))


def train_model(
    model_name: str,
    data_path: str,
    output_root: str,
    *,
    latent_dim: int | None = None,
    hidden_dim: int | None = None,
    embedding_dim: int | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    max_length: int | None = None,
    checkpoint_every: int = 50,
    seed: int | None = None,
) -> str:
    defaults = DEFAULTS[model_name]

    latent_dim = defaults.latent_dim if latent_dim is None else latent_dim
    hidden_dim = defaults.hidden_dim if hidden_dim is None else hidden_dim
    embedding_dim = defaults.embedding_dim if embedding_dim is None else embedding_dim
    batch_size = defaults.batch_size if batch_size is None else batch_size
    epochs = defaults.epochs if epochs is None else epochs
    learning_rate = defaults.learning_rate if learning_rate is None else learning_rate
    max_length = defaults.max_length if max_length is None else max_length

    if seed is not None:
        torch.manual_seed(seed)

    base_dir, checkpoints_dir = _make_run_dir(output_root, model_name)
    config_path = os.path.join(base_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": model_name,
                "data_path": data_path,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "embedding_dim": embedding_dim,
                "batch_size": batch_size,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "checkpoint_every": checkpoint_every,
                "seed": seed,
            },
            handle,
            indent=2,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_cls = dataset_class_for_model(model_name)
    dataset = dataset_cls(data_path, max_length)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty after parsing. Check grammar/data compatibility.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = build_model(
        model_name=model_name,
        dataset=dataset,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        max_length=max_length,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    checkpoint_meta = {
        "format_version": 2,
        "model_name": model_name,
        "max_length": max_length,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
    }
    if model_name == "char_vae":
        checkpoint_meta["vocab"] = dataset.vocabulary.vocab
    else:
        checkpoint_meta["num_productions"] = len(dataset.productions)

    for epoch in range(epochs):
        total_loss = 0.0
        total_ce = 0.0
        total_kld = 0.0
        beta = _beta_for_epoch(epoch, epochs)
        tf_ratio = _tf_ratio_for_epoch(epoch, epochs)

        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()

            if model_name == "grammar_vae_masked":
                recon_logits, mu, logvar = model(batch)
                loss, ce, kld = grammar_masked_vae_loss(
                    recon_logits,
                    batch,
                    mu,
                    logvar,
                    model.masks,
                    model.ind_of_ind,
                    beta=beta,
                )
            else:
                recon_logits, mu, logvar = model(batch, teacher_forcing_ratio=tf_ratio)
                pad_idx = dataset.pad_idx if model_name == "char_vae" else dataset.pad_rule_idx
                loss, ce, kld = sequence_vae_loss(recon_logits, batch, mu, logvar, pad_idx=pad_idx, beta=beta)
                if model_name == "vq_grammar_ae":
                    vq_loss = model.aux_loss() * batch.size(0)
                    loss = loss + vq_loss
                    kld = vq_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce.item()
            total_kld += kld.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(dataset)
            avg_ce = total_ce / len(dataset)
            avg_kld = total_kld / len(dataset)
            if model_name == "grammar_vae_masked":
                log_line = (
                    f"Epoch {epoch + 1}/{epochs}\tLoss: {avg_loss:.4f}\tCE: {avg_ce:.4f}"
                    f"\tKLD: {avg_kld:.4f}\tBeta: {beta:.2f}"
                )
            else:
                log_line = (
                    f"Epoch {epoch + 1}/{epochs}\tLoss: {avg_loss:.4f}\tCE: {avg_ce:.4f}"
                    f"\tKLD: {avg_kld:.4f}\tBeta: {beta:.2f}\tTF: {tf_ratio:.2f}"
                )
            print(log_line)
            with open(os.path.join(base_dir, "training.log"), "a", encoding="utf-8") as handle:
                handle.write(log_line + "\n")

        if (epoch + 1) % checkpoint_every == 0:
            torch.save(
                {"state_dict": model.state_dict(), "meta": checkpoint_meta},
                os.path.join(checkpoints_dir, f"epoch_{epoch + 1}.pth"),
            )

    final_path = os.path.join(base_dir, f"{model_name}.pth")
    torch.save({"state_dict": model.state_dict(), "meta": checkpoint_meta}, final_path)
    print(f"Saved model to: {final_path}")
    return final_path
