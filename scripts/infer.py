from __future__ import annotations

import argparse
import os
import random
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from f1vae.config.defaults import DEFAULTS
from f1vae.config.io import load_yaml
from f1vae.grammars import f1 as G
from f1vae.inference.checkpoints import load_checkpoint
from f1vae.inference.decode import decode_char_indices, decode_grammar_indices
from f1vae.inference.encode import encode_char_string, encode_grammar_onehot_string, encode_grammar_rule_string
from f1vae.models.char_vae import CharVAE
from f1vae.models.grammar_vae import GrammarRuleVAE
from f1vae.models.grammar_vae_masked import GrammarMaskedVAE
from f1vae.models.transformer_vae import TransformerGrammarVAE
from f1vae.models.tree_vae import TreeGrammarVAE
from f1vae.models.tree_vae_masked import MaskedTreeGrammarVAE
from f1vae.models.tree_vae_masked_lhs import LHSConditionedMaskedTreeGrammarVAE
from f1vae.models.vq_grammar_ae import VQGrammarAE
from f1vae.models.registry import MODEL_NAMES
from f1vae.utils import command_run_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-sample inference without full dataset parsing")
    parser.add_argument("--model", choices=MODEL_NAMES, required=False)
    parser.add_argument("--config-model", default=None, help="Path to model YAML config")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--mode", choices=("reconstruct", "mutate"), default="reconstruct")
    parser.add_argument("--input-string", default=None, help="Input genotype string")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    return parser


def _build_model(model_name: str, *, latent_dim: int, hidden_dim: int | None, embedding_dim: int | None, max_length: int, checkpoint_meta: dict):
    if model_name == "char_vae":
        vocab = checkpoint_meta.get("vocab")
        if not vocab:
            raise ValueError("Char model requires vocabulary in checkpoint metadata. Retrain with new pipeline.")
        return (
            CharVAE(
                vocab_size=len(vocab),
                emb_dim=int(embedding_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=latent_dim,
                max_length=max_length,
            ),
            vocab,
        )

    if model_name == "grammar_vae":
        num_rules = len(G.GCFG.productions())
        return (
            GrammarRuleVAE(
                num_classes=num_rules + 1,
                emb_dim=int(embedding_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=latent_dim,
                max_length=max_length,
                pad_rule_idx=num_rules,
            ),
            None,
        )

    if model_name == "grammar_vae_masked":
        return (GrammarMaskedVAE(latent_dim=latent_dim, max_length=max_length, n_chars=len(G.GCFG.productions())), None)

    if model_name == "tree_vae":
        num_rules = len(G.GCFG.productions())
        return (
            TreeGrammarVAE(
                num_classes=num_rules + 1,
                emb_dim=int(embedding_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=latent_dim,
                max_length=max_length,
                pad_rule_idx=num_rules,
            ),
            None,
        )

    if model_name == "tree_vae_masked":
        num_rules = len(G.GCFG.productions())
        return (
            MaskedTreeGrammarVAE(
                num_classes=num_rules + 1,
                emb_dim=int(embedding_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=latent_dim,
                max_length=max_length,
                pad_rule_idx=num_rules,
            ),
            None,
        )

    if model_name == "tree_vae_masked_lhs":
        num_rules = len(G.GCFG.productions())
        return (
            LHSConditionedMaskedTreeGrammarVAE(
                num_classes=num_rules + 1,
                emb_dim=int(embedding_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=latent_dim,
                max_length=max_length,
                pad_rule_idx=num_rules,
            ),
            None,
        )

    if model_name == "transformer_vae":
        num_rules = len(G.GCFG.productions())
        return (
            TransformerGrammarVAE(
                num_classes=num_rules + 1,
                emb_dim=int(embedding_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=latent_dim,
                max_length=max_length,
                pad_rule_idx=num_rules,
            ),
            None,
        )

    if model_name == "vq_grammar_ae":
        num_rules = len(G.GCFG.productions())
        return (
            VQGrammarAE(
                num_classes=num_rules + 1,
                emb_dim=int(embedding_dim),
                hidden_dim=int(hidden_dim),
                latent_dim=latent_dim,
                max_length=max_length,
                pad_rule_idx=num_rules,
            ),
            None,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    model_cfg = load_yaml(args.config_model)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, checkpoint_meta = load_checkpoint(args.weights, device)

    model_name = args.model or model_cfg.get("model") or checkpoint_meta.get("model_name")
    if model_name is None:
        raise ValueError("Model name is required. Provide --model, --config-model, or checkpoint metadata.")

    defaults = DEFAULTS[model_name]
    latent_dim = args.latent_dim or model_cfg.get("latent_dim") or checkpoint_meta.get("latent_dim") or defaults.latent_dim
    hidden_dim = args.hidden_dim or model_cfg.get("hidden_dim") or checkpoint_meta.get("hidden_dim") or defaults.hidden_dim
    embedding_dim = args.embedding_dim or model_cfg.get("embedding_dim") or checkpoint_meta.get("embedding_dim") or defaults.embedding_dim
    max_length = args.max_length or model_cfg.get("max_length") or checkpoint_meta.get("max_length") or defaults.max_length

    model, vocab = _build_model(
        model_name,
        latent_dim=int(latent_dim),
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        max_length=int(max_length),
        checkpoint_meta=checkpoint_meta,
    )
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.input_string is None:
        raise ValueError("--input-string is required for reconstruct/mutate modes")

    with torch.no_grad():
        if model_name == "char_vae":
            char2idx = {c: i for i, c in enumerate(vocab)}
            idx2char = {i: c for i, c in enumerate(vocab)}
            x = encode_char_string(args.input_string, char2idx, int(max_length)).to(device)
            mu, logvar = model.encoder(x)
            z = mu.clone()
            if args.mode == "mutate":
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                z[0, dim] += args.noise_scale * std[0, dim] * torch.randn(1, device=device).squeeze()
            generated = model.decode_from_latent(z).squeeze(0)
            out = decode_char_indices(generated, idx2char)
        elif model_name in {"grammar_vae", "transformer_vae"}:
            x = encode_grammar_rule_string(args.input_string, int(max_length)).to(device)
            mu, logvar = model.encoder(x)
            z = mu.clone()
            if args.mode == "mutate":
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                z[0, dim] += args.noise_scale * std[0, dim] * torch.randn(1, device=device).squeeze()
            logits = model.decoder(z, None, teacher_forcing_ratio=0.0).squeeze(0)
            out = decode_grammar_indices(logits.argmax(-1))
        elif model_name in {"tree_vae", "tree_vae_masked", "tree_vae_masked_lhs"}:
            x = encode_grammar_rule_string(args.input_string, int(max_length)).to(device)
            mu, logvar = model.encode(x)
            z = mu.clone()
            if args.mode == "mutate":
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                z[0, dim] += args.noise_scale * std[0, dim] * torch.randn(1, device=device).squeeze()
            logits = model.decode(z, None, teacher_forcing_ratio=0.0).squeeze(0)
            out = decode_grammar_indices(logits.argmax(-1))
        elif model_name == "vq_grammar_ae":
            x = encode_grammar_rule_string(args.input_string, int(max_length)).to(device)
            z = model.encoder(x)
            z = z.clone()
            if args.mode == "mutate":
                dim = random.randint(0, z.size(1) - 1)
                z[0, dim] += args.noise_scale * torch.randn(1, device=device).squeeze()
            logits = model.decoder(z, None, teacher_forcing_ratio=0.0).squeeze(0)
            out = decode_grammar_indices(logits.argmax(-1))
        else:
            x = encode_grammar_onehot_string(args.input_string, int(max_length)).to(device)
            mu, logvar = model.encoder(x)
            z = mu.clone()
            if args.mode == "mutate":
                dim = random.randint(0, z.size(1) - 1)
                std = torch.exp(0.5 * logvar)
                z[0, dim] += args.noise_scale * std[0, dim] * torch.randn(1, device=device).squeeze()
            generated = model.decode_masked(z).squeeze(0).argmax(-1)
            out = decode_grammar_indices(generated)

    print(f"Input : {args.input_string}")
    print(f"Output: {out}")


if __name__ == "__main__":
    with command_run_logger("scripts/infer.py"):
        main()
