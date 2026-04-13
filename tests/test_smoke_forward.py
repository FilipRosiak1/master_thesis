from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from f1vae.config.defaults import DEFAULTS
from f1vae.data.datasets import CharGenotypeDataset, GrammarOneHotDataset, GrammarRuleDataset
from f1vae.models.char_vae import CharVAE
from f1vae.models.grammar_vae import GrammarRuleVAE
from f1vae.models.grammar_vae_masked import GrammarMaskedVAE
from f1vae.training.losses import grammar_masked_vae_loss, sequence_vae_loss


def _write_tiny_dataset(path: Path) -> None:
    path.write_text("X\nRX\n", encoding="utf-8")


def test_char_vae_forward(tmp_path: Path):
    defaults = DEFAULTS["char_vae"]
    dataset_path = tmp_path / "tiny.txt"
    _write_tiny_dataset(dataset_path)

    dataset = CharGenotypeDataset(str(dataset_path), max_length=defaults.max_length)
    model = CharVAE(
        vocab_size=dataset.vocab_size,
        emb_dim=defaults.embedding_dim,
        hidden_dim=defaults.hidden_dim,
        latent_dim=defaults.latent_dim,
        max_length=defaults.max_length,
    )

    batch = dataset[0].unsqueeze(0)
    recon, mu, logvar = model(batch, teacher_forcing_ratio=0.0)
    loss, _, _ = sequence_vae_loss(recon, batch, mu, logvar, pad_idx=dataset.pad_idx)
    assert recon.shape[0] == 1
    assert mu.shape[-1] == defaults.latent_dim
    assert torch.isfinite(loss)


def test_grammar_rule_vae_forward(tmp_path: Path):
    defaults = DEFAULTS["grammar_vae"]
    dataset_path = tmp_path / "tiny.txt"
    _write_tiny_dataset(dataset_path)

    dataset = GrammarRuleDataset(str(dataset_path), max_length=defaults.max_length)
    model = GrammarRuleVAE(
        num_classes=dataset.num_classes,
        emb_dim=defaults.embedding_dim,
        hidden_dim=defaults.hidden_dim,
        latent_dim=defaults.latent_dim,
        max_length=defaults.max_length,
        pad_rule_idx=dataset.pad_rule_idx,
    )

    batch = dataset[0].unsqueeze(0)
    recon, mu, logvar = model(batch, teacher_forcing_ratio=0.0)
    loss, _, _ = sequence_vae_loss(recon, batch, mu, logvar, pad_idx=dataset.pad_rule_idx)
    assert recon.shape[0] == 1
    assert mu.shape[-1] == defaults.latent_dim
    assert torch.isfinite(loss)


def test_grammar_masked_vae_forward(tmp_path: Path):
    defaults = DEFAULTS["grammar_vae_masked"]
    dataset_path = tmp_path / "tiny.txt"
    _write_tiny_dataset(dataset_path)

    dataset = GrammarOneHotDataset(str(dataset_path), max_length=defaults.max_length)
    model = GrammarMaskedVAE(defaults.latent_dim, defaults.max_length, dataset.n_chars)

    batch = dataset[0].unsqueeze(0)
    recon, mu, logvar = model(batch)
    loss, _, _ = grammar_masked_vae_loss(recon, batch, mu, logvar, model.masks, model.ind_of_ind)
    assert recon.shape[0] == 1
    assert mu.shape[-1] == defaults.latent_dim
    assert torch.isfinite(loss)
