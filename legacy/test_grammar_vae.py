from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.config.defaults import DEFAULTS
from f1vae.data.datasets import GrammarRuleDataset
from f1vae.inference.evaluate import mutation_examples
from f1vae.models.grammar_vae import GrammarRuleVAE


def load_model(model_path: str, dataset_path: str):
    defaults = DEFAULTS["grammar_vae"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GrammarRuleDataset(dataset_path, defaults.max_length)
    model = GrammarRuleVAE(
        num_classes=dataset.num_classes,
        emb_dim=defaults.embedding_dim,
        hidden_dim=defaults.hidden_dim,
        latent_dim=defaults.latent_dim,
        max_length=defaults.max_length,
        pad_rule_idx=dataset.pad_rule_idx,
    ).to(device)
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, dataset, device


if __name__ == "__main__":
    custom_folder = ""
    checkpoint_name = "gvae_epoch_300.pth"
    base_models_dir = os.path.join(ROOT, "models", "f1", "grammar_vae")
    dataset_path = os.path.join(ROOT, "datasets", "f1", "f1_dataset.txt")

    selected_folder = os.path.join(base_models_dir, custom_folder) if custom_folder else sorted(
        [f for f in os.listdir(base_models_dir) if os.path.isdir(os.path.join(base_models_dir, f))]
    )[-1]
    if not os.path.isabs(selected_folder):
        selected_folder = os.path.join(base_models_dir, selected_folder)

    model_path = (
        os.path.join(selected_folder, "checkpoints", checkpoint_name)
        if checkpoint_name
        else os.path.join(selected_folder, "grammar_vae.pth")
    )

    model, dataset, device = load_model(model_path, dataset_path)
    examples = mutation_examples("grammar_vae", model, dataset, device, num_samples=10, noise_scale=1.0)
    for i, (original, mutated) in enumerate(examples, start=1):
        print(f"Sample {i:02d}.")
        print(f"  Original: {original}")
        print(f"  Mutated : {mutated}")
