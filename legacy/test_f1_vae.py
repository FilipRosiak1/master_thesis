from __future__ import annotations

import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.config.defaults import DEFAULTS
from f1vae.data.datasets import CharGenotypeDataset
from f1vae.inference.decode import decode_char_indices
from f1vae.models.char_vae import CharVAE


def load_model(model_path: str, dataset_path: str):
    defaults = DEFAULTS["char_vae"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CharGenotypeDataset(dataset_path, max_length=defaults.max_length)
    model = CharVAE(
        vocab_size=dataset.vocab_size,
        emb_dim=defaults.embedding_dim,
        hidden_dim=defaults.hidden_dim,
        latent_dim=defaults.latent_dim,
        max_length=defaults.max_length,
    ).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, dataset, device


def mutate_existing_genotypes(model, dataset, device, num_samples: int = 10, noise_scale: float = 1.0):
    print(f"\n--- Mutating {num_samples} Random Genotypes (noise scale: {noise_scale}) ---")

    for i in range(num_samples):
        original_idx = random.randint(0, len(dataset) - 1)
        original_tensor = dataset[original_idx]
        original_str = decode_char_indices(original_tensor, dataset.vocabulary.idx2char)

        original_tensor = original_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            mu, logvar = model.encoder(original_tensor)
            z = mu.clone()
            dim_to_mutate = random.randint(0, z.size(1) - 1)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(1, device=device).squeeze()
            z[0, dim_to_mutate] += noise_scale * std[0, dim_to_mutate] * eps
            generated_indices = model.decode_from_latent(z).squeeze(0)

        mutated_str = decode_char_indices(generated_indices, dataset.vocabulary.idx2char)
        print(f"Sample {i + 1:02d}.")
        print(f"  Original: {original_str}")
        print(f"  Mutated : {mutated_str}")


if __name__ == "__main__":
    custom_folder = "2026-02-24_03-46-47"
    checkpoint_name = "vae_f1_genotype_epoch_150.pth"
    base_models_dir = os.path.join(ROOT, "models", "f1", "vae")
    dataset_path = os.path.join(ROOT, "datasets", "f1", "f1_dataset.txt")

    selected_folder = os.path.join(base_models_dir, custom_folder) if custom_folder else sorted(
        [f for f in os.listdir(base_models_dir) if os.path.isdir(os.path.join(base_models_dir, f))]
    )[-1]
    if not os.path.isabs(selected_folder):
        selected_folder = os.path.join(base_models_dir, selected_folder)

    model_path = (
        os.path.join(selected_folder, "checkpoints", checkpoint_name)
        if checkpoint_name
        else os.path.join(selected_folder, "char_vae.pth")
    )

    model, dataset, device = load_model(model_path, dataset_path)
    mutate_existing_genotypes(model, dataset, device, num_samples=10, noise_scale=0.0)
