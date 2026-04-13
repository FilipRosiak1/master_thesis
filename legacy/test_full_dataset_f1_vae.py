from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.config.defaults import DEFAULTS
from f1vae.data.datasets import CharGenotypeDataset
from f1vae.inference.evaluate import reconstruction_metrics
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
    metrics = reconstruction_metrics("char_vae", model, dataset, device)
    print("\nResults:")
    print(f"  Total Genotypes: {metrics['total']}")
    print(f"  Exactly Reconstructed: {metrics['exact']}")
    print(f"  Exact Accuracy: {metrics['exact_accuracy']:.2f}%")
    print(f"  Average Similarity: {metrics['avg_similarity']:.2f}%")
