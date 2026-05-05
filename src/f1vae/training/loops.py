from __future__ import annotations

import json
import os
import time
import difflib
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from f1vae.config.defaults import DEFAULTS
from f1vae.inference.decode import decode_char_indices, decode_grammar_indices, decode_masked_deterministic
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


def _format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _reconstruct_strings_from_batch(model_name: str, model, batch: torch.Tensor, idx2char: dict[int, str] | None) -> list[str]:
    if model_name == "char_vae":
        if idx2char is None:
            raise ValueError("idx2char is required for char_vae reconstruction metrics")
        mu, _ = model.encoder(batch)
        generated = model.decode_from_latent(mu)
        return [decode_char_indices(generated[i], idx2char) for i in range(generated.size(0))]

    if model_name == "grammar_vae_masked":
        mu, _ = model.encoder(batch)
        generated = decode_masked_deterministic(model, mu)
        return [decode_grammar_indices(generated[i]) for i in range(generated.size(0))]

    if model_name in {"tree_vae", "tree_vae_masked"}:
        mu, _ = model.encode(batch)
        logits = model.decode(mu, None, teacher_forcing_ratio=0.0)
        generated = logits.argmax(dim=-1)
        return [decode_grammar_indices(generated[i]) for i in range(generated.size(0))]

    if model_name == "vq_grammar_ae":
        z = model.encoder(batch)
        logits = model.decoder(z, None, teacher_forcing_ratio=0.0)
        generated = logits.argmax(dim=-1)
        return [decode_grammar_indices(generated[i]) for i in range(generated.size(0))]

    mu, _ = model.encoder(batch)
    logits = model.decoder(mu, None, teacher_forcing_ratio=0.0)
    generated = logits.argmax(dim=-1)
    return [decode_grammar_indices(generated[i]) for i in range(generated.size(0))]


def _reference_strings_from_batch(model_name: str, batch: torch.Tensor, idx2char: dict[int, str] | None) -> list[str]:
    if model_name == "char_vae":
        if idx2char is None:
            raise ValueError("idx2char is required for char_vae reference decoding")
        return [decode_char_indices(batch[i], idx2char) for i in range(batch.size(0))]

    if model_name == "grammar_vae_masked":
        targets = batch.argmax(dim=-1)
        return [decode_grammar_indices(targets[i]) for i in range(targets.size(0))]

    return [decode_grammar_indices(batch[i]) for i in range(batch.size(0))]


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
    val_split: float = 0.0,
    val_every: int = 10,
    schedule_epochs: int | None = None,
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

    if val_split < 0.0 or val_split >= 1.0:
        raise ValueError("val_split must be in range [0.0, 1.0).")
    if val_every < 1:
        raise ValueError("val_every must be >= 1.")
    if schedule_epochs is not None and schedule_epochs < 1:
        raise ValueError("schedule_epochs must be >= 1 when provided.")
    schedule_epochs = epochs if schedule_epochs is None else schedule_epochs

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
                "val_split": val_split,
                "val_every": val_every,
                "schedule_epochs": schedule_epochs,
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

    val_loader = None
    train_dataset_len = len(dataset)
    val_dataset_len = 0
    if val_split > 0.0 and len(dataset) > 1:
        val_size = int(len(dataset) * val_split)
        val_size = max(1, val_size)
        val_size = min(len(dataset) - 1, val_size)
        train_size = len(dataset) - val_size
        split_generator = torch.Generator().manual_seed(seed) if seed is not None else None
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        train_dataset_len = len(train_dataset)
        val_dataset_len = len(val_dataset)
    else:
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
    train_start = time.perf_counter()
    best_val_exact = -1
    best_val_similarity = -1.0

    checkpoint_meta = {
        "format_version": 2,
        "model_name": model_name,
        "data_path": data_path,
        "max_length": max_length,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
    }
    if model_name == "char_vae":
        checkpoint_meta["vocab"] = dataset.vocabulary.vocab
        idx2char = dataset.vocabulary.idx2char
    else:
        checkpoint_meta["num_productions"] = len(dataset.productions)
        idx2char = None

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        total_ce = 0.0
        total_kld = 0.0
        beta = _beta_for_epoch(epoch, schedule_epochs)
        tf_ratio = _tf_ratio_for_epoch(epoch, schedule_epochs)

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
            elif model_name in {"tree_vae", "tree_vae_masked"}:
                recon_logits, mu, logvar = model(batch, teacher_forcing_ratio=tf_ratio)
                loss, ce, kld = sequence_vae_loss(recon_logits, batch, mu, logvar, pad_idx=dataset.pad_rule_idx, beta=beta)
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
            avg_loss = total_loss / train_dataset_len
            avg_ce = total_ce / train_dataset_len
            avg_kld = total_kld / train_dataset_len
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

        if val_loader is not None and ((epoch + 1) % max(1, val_every) == 0 or epoch == 0 or (epoch + 1) == epochs):
            model.eval()
            val_total_loss = 0.0
            val_total_ce = 0.0
            val_total_kld = 0.0
            val_exact = 0
            val_similarity_sum = 0.0
            val_recon_total = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(device)
                    if model_name == "grammar_vae_masked":
                        val_recon_logits, val_mu, val_logvar = model(val_batch)
                        val_loss, val_ce, val_kld = grammar_masked_vae_loss(
                            val_recon_logits,
                            val_batch,
                            val_mu,
                            val_logvar,
                            model.masks,
                            model.ind_of_ind,
                            beta=beta,
                        )
                    elif model_name in {"tree_vae", "tree_vae_masked"}:
                        val_recon_logits, val_mu, val_logvar = model(val_batch, teacher_forcing_ratio=0.0)
                        val_loss, val_ce, val_kld = sequence_vae_loss(
                            val_recon_logits,
                            val_batch,
                            val_mu,
                            val_logvar,
                            pad_idx=dataset.pad_rule_idx,
                            beta=beta,
                        )
                    else:
                        val_recon_logits, val_mu, val_logvar = model(val_batch, teacher_forcing_ratio=0.0)
                        val_pad_idx = dataset.pad_idx if model_name == "char_vae" else dataset.pad_rule_idx
                        val_loss, val_ce, val_kld = sequence_vae_loss(
                            val_recon_logits,
                            val_batch,
                            val_mu,
                            val_logvar,
                            pad_idx=val_pad_idx,
                            beta=beta,
                        )
                        if model_name == "vq_grammar_ae":
                            val_vq_loss = model.aux_loss() * val_batch.size(0)
                            val_loss = val_loss + val_vq_loss
                            val_kld = val_vq_loss

                    val_total_loss += val_loss.item()
                    val_total_ce += val_ce.item()
                    val_total_kld += val_kld.item()

                    originals = _reference_strings_from_batch(model_name, val_batch, idx2char)
                    reconstructions = _reconstruct_strings_from_batch(model_name, model, val_batch, idx2char)
                    for original, reconstructed in zip(originals, reconstructions):
                        if original == reconstructed:
                            val_exact += 1
                        val_similarity_sum += difflib.SequenceMatcher(None, original, reconstructed).ratio()
                        val_recon_total += 1

            val_line = (
                f"Val {epoch + 1}/{epochs}\tLoss: {val_total_loss / val_dataset_len:.4f}"
                f"\tCE: {val_total_ce / val_dataset_len:.4f}\tKLD: {val_total_kld / val_dataset_len:.4f}"
            )
            print(val_line)
            with open(os.path.join(base_dir, "training.log"), "a", encoding="utf-8") as handle:
                handle.write(val_line + "\n")

            val_exact_acc = val_exact / max(1, val_recon_total)
            val_avg_similarity = val_similarity_sum / max(1, val_recon_total)
            val_recon_line = (
                f"ValRecon {epoch + 1}/{epochs}\tExact: {val_exact}/{val_recon_total}"
                f"\tExactAcc: {(100.0 * val_exact_acc):.2f}%"
                f"\tAvgSim: {(100.0 * val_avg_similarity):.2f}%"
            )
            print(val_recon_line)
            with open(os.path.join(base_dir, "training.log"), "a", encoding="utf-8") as handle:
                handle.write(val_recon_line + "\n")

            if val_exact > best_val_exact or (
                val_exact == best_val_exact and val_avg_similarity > best_val_similarity
            ):
                best_val_exact = val_exact
                best_val_similarity = val_avg_similarity
                best_meta = {
                    **checkpoint_meta,
                    "best_epoch": epoch + 1,
                    "val_exact": val_exact,
                    "val_total": val_recon_total,
                    "val_exact_acc": val_exact_acc,
                    "val_avg_similarity": val_avg_similarity,
                }
                torch.save(
                    {"state_dict": model.state_dict(), "meta": best_meta},
                    os.path.join(base_dir, "best_val_recon.pth"),
                )
                best_line = (
                    f"BestValRecon {epoch + 1}/{epochs}\tExact: {val_exact}/{val_recon_total}"
                    f"\tExactAcc: {(100.0 * val_exact_acc):.2f}%"
                    f"\tAvgSim: {(100.0 * val_avg_similarity):.2f}%"
                )
                print(best_line)
                with open(os.path.join(base_dir, "training.log"), "a", encoding="utf-8") as handle:
                    handle.write(best_line + "\n")
            model.train()

        elapsed = time.perf_counter() - train_start
        avg_epoch_time = elapsed / (epoch + 1)
        remaining = avg_epoch_time * (epochs - (epoch + 1))
        if (epoch + 1) % 10 == 0 or epoch == 0:
            eta_line = (
                f"Progress {epoch + 1}/{epochs}"
                f"\tEpoch time: {_format_duration(time.perf_counter() - epoch_start)}"
                f"\tElapsed: {_format_duration(elapsed)}"
                f"\tETA: {_format_duration(remaining)}"
            )
            print(eta_line)
            with open(os.path.join(base_dir, "training.log"), "a", encoding="utf-8") as handle:
                handle.write(eta_line + "\n")

    final_path = os.path.join(base_dir, f"{model_name}.pth")
    torch.save({"state_dict": model.state_dict(), "meta": checkpoint_meta}, final_path)
    print(f"Saved model to: {final_path}")
    return final_path
