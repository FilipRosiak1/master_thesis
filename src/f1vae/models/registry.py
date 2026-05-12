from __future__ import annotations

from f1vae.data.datasets import CharGenotypeDataset, GrammarOneHotDataset, GrammarRuleDataset
from f1vae.models import (
    CharVAE,
    GrammarMaskedVAE,
    GrammarRuleVAE,
    LHSConditionedMaskedTreeGrammarVAE,
    LHSDepthConditionedMaskedTreeGrammarVAE,
    MaskedTreeGrammarVAE,
    TransformerGrammarVAE,
    TreeGrammarVAE,
    VQGrammarAE,
)


MODEL_NAMES = (
    "char_vae",
    "grammar_vae",
    "grammar_vae_masked",
    "tree_vae",
    "tree_vae_masked",
    "tree_vae_masked_lhs",
    "tree_vae_masked_lhs_depth",
    "transformer_vae",
    "vq_grammar_ae",
)


def dataset_class_for_model(model_name: str):
    if model_name == "char_vae":
        return CharGenotypeDataset
    if model_name == "grammar_vae":
        return GrammarRuleDataset
    if model_name == "grammar_vae_masked":
        return GrammarOneHotDataset
    if model_name in {
        "tree_vae",
        "tree_vae_masked",
        "tree_vae_masked_lhs",
        "tree_vae_masked_lhs_depth",
        "transformer_vae",
        "vq_grammar_ae",
    }:
        return GrammarRuleDataset
    raise ValueError(f"Unsupported model: {model_name}")


def build_model(model_name: str, dataset, latent_dim: int, hidden_dim: int | None, embedding_dim: int | None, max_length: int):
    if model_name == "char_vae":
        return CharVAE(
            vocab_size=dataset.vocab_size,
            emb_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=latent_dim,
            max_length=max_length,
        )

    if model_name == "grammar_vae":
        return GrammarRuleVAE(
            num_classes=dataset.num_classes,
            emb_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=dataset.pad_rule_idx,
        )

    if model_name == "grammar_vae_masked":
        return GrammarMaskedVAE(
            latent_dim=latent_dim,
            max_length=max_length,
            n_chars=dataset.n_chars,
        )

    if model_name == "tree_vae":
        return TreeGrammarVAE(
            num_classes=dataset.num_classes,
            emb_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=dataset.pad_rule_idx,
        )

    if model_name == "tree_vae_masked":
        return MaskedTreeGrammarVAE(
            num_classes=dataset.num_classes,
            emb_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=dataset.pad_rule_idx,
        )

    if model_name == "tree_vae_masked_lhs":
        return LHSConditionedMaskedTreeGrammarVAE(
            num_classes=dataset.num_classes,
            emb_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=dataset.pad_rule_idx,
        )

    if model_name == "tree_vae_masked_lhs_depth":
        return LHSDepthConditionedMaskedTreeGrammarVAE(
            num_classes=dataset.num_classes,
            emb_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=dataset.pad_rule_idx,
        )

    if model_name == "transformer_vae":
        return TransformerGrammarVAE(
            num_classes=dataset.num_classes,
            emb_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=dataset.pad_rule_idx,
        )

    if model_name == "vq_grammar_ae":
        return VQGrammarAE(
            num_classes=dataset.num_classes,
            emb_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=dataset.pad_rule_idx,
        )

    raise ValueError(f"Unsupported model: {model_name}")
