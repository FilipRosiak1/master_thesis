from __future__ import annotations

import difflib
import random

import torch
from torch.utils.data import DataLoader, Subset

from f1vae.inference.decode import decode_char_indices, decode_grammar_indices, decode_masked_deterministic


def reconstruction_metrics(
    model_name: str,
    model,
    dataset,
    device: torch.device,
    max_items: int | None = None,
    batch_size: int = 64,
) -> dict[str, float]:
    total = len(dataset) if max_items is None else min(len(dataset), max_items)
    if total == 0:
        return {"total": 0, "exact": 0, "exact_accuracy": 0.0, "avg_similarity": 0.0}

    batch_size = max(1, int(batch_size))
    loader = DataLoader(Subset(dataset, range(total)), batch_size=batch_size, shuffle=False)

    exact = 0
    similarity_sum = 0.0
    processed = 0

    with torch.no_grad():
        for batch in loader:
            input_tensor = batch.to(device)
            current_batch = input_tensor.size(0)

            if model_name == "char_vae":
                mu, _ = model.encoder(input_tensor)
                generated_batch = model.decode_from_latent(mu)
                for row in range(current_batch):
                    original = decode_char_indices(batch[row], dataset.vocabulary.idx2char)
                    reconstructed = decode_char_indices(generated_batch[row], dataset.vocabulary.idx2char)
                    if original == reconstructed:
                        exact += 1
                    similarity_sum += difflib.SequenceMatcher(None, original, reconstructed).ratio()
            elif model_name in {"grammar_vae", "transformer_vae"}:
                mu, _ = model.encoder(input_tensor)
                logits_batch = model.decoder(mu, None, teacher_forcing_ratio=0.0)
                generated_batch = logits_batch.argmax(-1)
                for row in range(current_batch):
                    original = dataset.valid_lines[processed + row]
                    reconstructed = decode_grammar_indices(generated_batch[row])
                    if original == reconstructed:
                        exact += 1
                    similarity_sum += difflib.SequenceMatcher(None, original, reconstructed).ratio()
            elif model_name in {"tree_vae", "tree_vae_masked", "tree_vae_masked_lhs", "tree_vae_masked_lhs_depth"}:
                mu, _ = model.encode(input_tensor)
                logits_batch = model.decode(mu, None, teacher_forcing_ratio=0.0)
                generated_batch = logits_batch.argmax(-1)
                for row in range(current_batch):
                    original = dataset.valid_lines[processed + row]
                    reconstructed = decode_grammar_indices(generated_batch[row])
                    if original == reconstructed:
                        exact += 1
                    similarity_sum += difflib.SequenceMatcher(None, original, reconstructed).ratio()
            elif model_name == "vq_grammar_ae":
                z = model.encoder(input_tensor)
                logits_batch = model.decoder(z, None, teacher_forcing_ratio=0.0)
                generated_batch = logits_batch.argmax(-1)
                for row in range(current_batch):
                    original = dataset.valid_lines[processed + row]
                    reconstructed = decode_grammar_indices(generated_batch[row])
                    if original == reconstructed:
                        exact += 1
                    similarity_sum += difflib.SequenceMatcher(None, original, reconstructed).ratio()
            elif model_name == "grammar_vae_masked":
                mu, _ = model.encoder(input_tensor)
                generated_batch = decode_masked_deterministic(model, mu)
                for row in range(current_batch):
                    original = dataset.valid_lines[processed + row]
                    reconstructed = decode_grammar_indices(generated_batch[row])
                    if original == reconstructed:
                        exact += 1
                    similarity_sum += difflib.SequenceMatcher(None, original, reconstructed).ratio()
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            processed += current_batch

    return {
        "total": total,
        "exact": exact,
        "exact_accuracy": (exact / total) * 100.0,
        "avg_similarity": (similarity_sum / total) * 100.0,
    }


def mutation_examples(
    model_name: str,
    model,
    dataset,
    device: torch.device,
    *,
    num_samples: int,
    noise_scale: float,
) -> list[tuple[str, str]]:
    if len(dataset) == 0:
        return []

    examples = []
    with torch.no_grad():
        for _ in range(num_samples):
            idx = random.randint(0, len(dataset) - 1)
            sample = dataset[idx]
            input_tensor = sample.unsqueeze(0).to(device)

            if model_name == "char_vae":
                original = decode_char_indices(sample, dataset.vocabulary.idx2char)
                mu, logvar = model.encoder(input_tensor)
                z = mu.clone()
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(1, device=device).squeeze()
                z[0, dim] += noise_scale * std[0, dim] * eps
                generated = model.decode_from_latent(z).squeeze(0)
                mutated = decode_char_indices(generated, dataset.vocabulary.idx2char)
            elif model_name in {"grammar_vae", "transformer_vae"}:
                original = dataset.valid_lines[idx]
                mu, logvar = model.encoder(input_tensor)
                z = mu.clone()
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(1, device=device).squeeze()
                z[0, dim] += noise_scale * std[0, dim] * eps
                logits = model.decoder(z, None, teacher_forcing_ratio=0.0).squeeze(0)
                mutated = decode_grammar_indices(logits.argmax(-1))
            elif model_name in {"tree_vae", "tree_vae_masked", "tree_vae_masked_lhs", "tree_vae_masked_lhs_depth"}:
                original = dataset.valid_lines[idx]
                mu, logvar = model.encode(input_tensor)
                z = mu.clone()
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(1, device=device).squeeze()
                z[0, dim] += noise_scale * std[0, dim] * eps
                logits = model.decode(z, None, teacher_forcing_ratio=0.0).squeeze(0)
                mutated = decode_grammar_indices(logits.argmax(-1))
            elif model_name == "vq_grammar_ae":
                original = dataset.valid_lines[idx]
                z = model.encoder(input_tensor)
                z = z.clone()
                dim = random.randint(0, z.size(1) - 1)
                z[0, dim] += noise_scale * torch.randn(1, device=device).squeeze()
                logits = model.decoder(z, None, teacher_forcing_ratio=0.0).squeeze(0)
                mutated = decode_grammar_indices(logits.argmax(-1))
            elif model_name == "grammar_vae_masked":
                original = dataset.valid_lines[idx]
                mu, logvar = model.encoder(input_tensor)
                z = mu.clone()
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(1, device=device).squeeze()
                z[0, dim] += noise_scale * std[0, dim] * eps
                generated = model.decode_masked(z).squeeze(0).argmax(-1)
                mutated = decode_grammar_indices(generated)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            examples.append((original, mutated))

    return examples
