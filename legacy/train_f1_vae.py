"""Legacy compatibility entrypoint for character VAE training."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.config.defaults import DEFAULTS
from f1vae.data.datasets import CharGenotypeDataset
from f1vae.models.char_vae import CharVAE
from f1vae.training.loops import train_model

_DEFAULT = DEFAULTS["char_vae"]
LATENT_DIM = _DEFAULT.latent_dim
HIDDEN_DIM = _DEFAULT.hidden_dim
EMBEDDING_DIM = _DEFAULT.embedding_dim
BATCH_SIZE = _DEFAULT.batch_size
EPOCHS = _DEFAULT.epochs
LEARNING_RATE = _DEFAULT.learning_rate
MAX_LENGTH = _DEFAULT.max_length


class GenotypeDataset(CharGenotypeDataset):
    def __init__(self, filepath: str):
        super().__init__(filepath=filepath, max_length=MAX_LENGTH)
        self.vocab = self.vocabulary.vocab
        self.char2idx = self.vocabulary.char2idx
        self.idx2char = self.vocabulary.idx2char


class GenotypeVAE(CharVAE):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_length=MAX_LENGTH,
        )


def train() -> str:
    return train_model(
        model_name="char_vae",
        data_path=os.path.join(ROOT, "datasets", "f1", "f1_dataset.txt"),
        output_root=os.path.join(ROOT, "models", "f1"),
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        max_length=MAX_LENGTH,
    )


if __name__ == "__main__":
    train()
