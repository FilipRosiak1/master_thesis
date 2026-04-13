"""Legacy compatibility entrypoint for masked grammar VAE training."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.config.defaults import DEFAULTS
from f1vae.data.datasets import GrammarOneHotDataset as GrammarGenotypeDataset
from f1vae.models.grammar_vae_masked import GrammarMaskedVAE as GrammarVAE
from f1vae.training.loops import train_model

_DEFAULT = DEFAULTS["grammar_vae_masked"]
LATENT_DIM = _DEFAULT.latent_dim
MAX_LENGTH = _DEFAULT.max_length
BATCH_SIZE = _DEFAULT.batch_size
EPOCHS = _DEFAULT.epochs
LEARNING_RATE = _DEFAULT.learning_rate


def train() -> str:
    return train_model(
        model_name="grammar_vae_masked",
        data_path=os.path.join(ROOT, "datasets", "f1", "f1_dataset.txt"),
        output_root=os.path.join(ROOT, "models", "f1"),
        latent_dim=LATENT_DIM,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
    )


if __name__ == "__main__":
    train()
