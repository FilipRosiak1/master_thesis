"""Legacy compatibility entrypoint for rule-index grammar VAE training."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.config.defaults import DEFAULTS
from f1vae.data.datasets import GrammarRuleDataset
from f1vae.grammars import f1 as _G
from f1vae.models.grammar_vae import GrammarRuleVAE
from f1vae.training.loops import train_model

_DEFAULT = DEFAULTS["grammar_vae"]
LATENT_DIM = _DEFAULT.latent_dim
HIDDEN_DIM = _DEFAULT.hidden_dim
EMBEDDING_DIM = _DEFAULT.embedding_dim
BATCH_SIZE = _DEFAULT.batch_size
EPOCHS = _DEFAULT.epochs
LEARNING_RATE = _DEFAULT.learning_rate
MAX_LENGTH = _DEFAULT.max_length

f1_cfg = _G.GCFG
rules_list = f1_cfg.productions()
rule2idx = {rule: i for i, rule in enumerate(rules_list)}
idx2rule = {i: rule for i, rule in enumerate(rules_list)}
num_rules = len(rules_list)
PAD_RULE_IDX = num_rules
num_classes = num_rules + 1


class GrammarGenotypeDataset(GrammarRuleDataset):
    """Backward-compatible dataset signature used by existing tests."""

    def __init__(self, filepath: str, grammar=None, _rule2idx=None, max_len: int = MAX_LENGTH) -> None:
        super().__init__(filepath=filepath, max_length=max_len)


class GrammarGenotypeVAE(GrammarRuleVAE):
    def __init__(self, num_classes_: int, emb_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__(
            num_classes=num_classes_,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_length=MAX_LENGTH,
            pad_rule_idx=PAD_RULE_IDX,
        )


def train_gvae() -> str:
    return train_model(
        model_name="grammar_vae",
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
    train_gvae()
