from __future__ import annotations

import difflib
import random

import torch

from f1vae.inference.decode import decode_char_indices, decode_grammar_indices, decode_masked_deterministic


def reconstruction_metrics(model_name: str, model, dataset, device: torch.device, max_items: int | None = None) -> dict[str, float]:
    total = len(dataset) if max_items is None else min(len(dataset), max_items)
    if total == 0:
        return {"total": 0, "exact": 0, "exact_accuracy": 0.0, "avg_similarity": 0.0}

    exact = 0
    similarity_sum = 0.0

    with torch.no_grad():
        for i in range(total):
            sample = dataset[i]
            input_tensor = sample.unsqueeze(0).to(device)

            if model_name == "char_vae":
                original = decode_char_indices(sample, dataset.vocabulary.idx2char)
                mu, _ = model.encoder(input_tensor)
                generated = model.decode_from_latent(mu).squeeze(0)
                reconstructed = decode_char_indices(generated, dataset.vocabulary.idx2char)
            elif model_name in {"grammar_vae", "tree_vae", "transformer_vae"}:
                original = dataset.valid_lines[i]
                mu, _ = model.encoder(input_tensor)
                logits = model.decoder(mu, None, teacher_forcing_ratio=0.0).squeeze(0)
                generated = logits.argmax(-1)
                reconstructed = decode_grammar_indices(generated)
            elif model_name == "vq_grammar_ae":
                original = dataset.valid_lines[i]
                z = model.encoder(input_tensor)
                logits = model.decoder(z, None, teacher_forcing_ratio=0.0).squeeze(0)
                generated = logits.argmax(-1)
                reconstructed = decode_grammar_indices(generated)
            elif model_name == "grammar_vae_masked":
                original = dataset.valid_lines[i]
                mu, _ = model.encoder(input_tensor)
                generated = decode_masked_deterministic(model, mu).squeeze(0)
                reconstructed = decode_grammar_indices(generated)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            if original == reconstructed:
                exact += 1
            similarity_sum += difflib.SequenceMatcher(None, original, reconstructed).ratio()

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
            elif model_name in {"grammar_vae", "tree_vae", "transformer_vae"}:
                original = dataset.valid_lines[idx]
                mu, logvar = model.encoder(input_tensor)
                z = mu.clone()
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn(1, device=device).squeeze()
                z[0, dim] += noise_scale * std[0, dim] * eps
                logits = model.decoder(z, None, teacher_forcing_ratio=0.0).squeeze(0)
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
