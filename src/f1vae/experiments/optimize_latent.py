from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from f1vae.config.defaults import DEFAULTS
from f1vae.config.io import load_yaml
from f1vae.grammars import f1 as G
from f1vae.inference.checkpoints import load_checkpoint
from f1vae.inference.decode import decode_char_indices, decode_grammar_indices, decode_masked_deterministic
from f1vae.models.char_vae import CharVAE
from f1vae.models.grammar_vae import GrammarRuleVAE
from f1vae.models.grammar_vae_masked import GrammarMaskedVAE
from f1vae.models.registry import MODEL_NAMES
from f1vae.models.transformer_vae import TransformerGrammarVAE
from f1vae.models.tree_vae import TreeGrammarVAE
from f1vae.models.tree_vae_masked import MaskedTreeGrammarVAE
from f1vae.models.tree_vae_masked_lhs import (
    LHSConditionedMaskedTreeGrammarVAE,
    LHSDepthConditionedMaskedTreeGrammarVAE,
)
from f1vae.models.vq_grammar_ae import VQGrammarAE
from f1vae.optimization import CEMConfig, CMAESConfig, optimize_latent_cem, optimize_latent_cmaes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimize latent vector z for higher fitness")
    parser.add_argument("--model", choices=MODEL_NAMES, required=False)
    parser.add_argument("--config-model", default=None, help="Path to model YAML config")
    parser.add_argument("--weights", required=True, help="Path to trained .pth")
    parser.add_argument("--fitness-fn", required=True, help="Fitness function target: module:function or path.py:function")
    parser.add_argument("--algorithm", choices=("cmaes", "cem"), default="cmaes")
    parser.add_argument(
        "--cma-backend",
        choices=("auto", "internal"),
        default="auto",
        help="CMA-ES backend: auto uses python-cma if installed, otherwise internal",
    )
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--elite-fraction", type=float, default=0.2)
    parser.add_argument("--initial-sigma", type=float, default=0.8)
    parser.add_argument("--initial-std", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=0.2)
    parser.add_argument("--min-std", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    return parser


def _load_fitness_callable(target: str) -> Callable[[str], float]:
    if ":" not in target:
        raise ValueError("--fitness-fn must use format module:function or path.py:function")

    module_ref, func_name = target.split(":", 1)

    if module_ref.endswith(".py"):
        path = Path(module_ref).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Fitness module file does not exist: {path}")
        spec = importlib.util.spec_from_file_location("_user_fitness_module", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Could not load fitness module from: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)

    if not hasattr(module, func_name):
        raise AttributeError(f"Function '{func_name}' not found in '{module_ref}'")

    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"'{func_name}' is not callable")

    signature = inspect.signature(fn)
    if len(signature.parameters) == 0:
        raise TypeError("Fitness function must accept at least one argument: genotype string")

    return fn


def _build_model(
    model_name: str,
    *,
    latent_dim: int,
    hidden_dim: int | None,
    embedding_dim: int | None,
    max_length: int,
    checkpoint_meta: dict,
):
    if model_name == "char_vae":
        vocab = checkpoint_meta.get("vocab")
        if not vocab:
            raise ValueError("Char model requires checkpoint metadata field 'vocab'.")
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

    num_rules = len(G.GCFG.productions())
    if model_name == "grammar_vae":
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
        return (
            GrammarMaskedVAE(latent_dim=latent_dim, max_length=max_length, n_chars=num_rules),
            None,
        )
    if model_name == "tree_vae":
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
    if model_name == "tree_vae_masked_lhs_depth":
        return (
            LHSDepthConditionedMaskedTreeGrammarVAE(
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


def _decode_from_z(model_name: str, model, z: np.ndarray, device: torch.device, idx2char: dict[int, str] | None) -> str:
    z_tensor = torch.tensor(z, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if model_name == "char_vae":
            if idx2char is None:
                raise ValueError("idx2char is required for char_vae decoding")
            tokens = model.decode_from_latent(z_tensor).squeeze(0)
            return decode_char_indices(tokens, idx2char)
        if model_name == "grammar_vae_masked":
            tokens = decode_masked_deterministic(model, z_tensor).squeeze(0)
            return decode_grammar_indices(tokens)
        if model_name in {"tree_vae_masked", "tree_vae_masked_lhs", "tree_vae_masked_lhs_depth"}:
            logits = model.decode(z_tensor, None, teacher_forcing_ratio=0.0).squeeze(0)
            return decode_grammar_indices(logits.argmax(dim=-1))
        logits = model.decoder(z_tensor, None, teacher_forcing_ratio=0.0).squeeze(0)
        return decode_grammar_indices(logits.argmax(dim=-1))


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

    idx2char = None
    if vocab is not None:
        idx2char = {i: c for i, c in enumerate(vocab)}

    fitness_fn = _load_fitness_callable(args.fitness_fn)

    best_genotype = ""
    best_score = -np.inf

    def objective(z: np.ndarray) -> float:
        nonlocal best_genotype
        nonlocal best_score
        genotype = _decode_from_z(model_name, model, z, device, idx2char)
        score = float(fitness_fn(genotype))
        if score > best_score:
            best_score = score
            best_genotype = genotype
        return score

    if args.algorithm == "cmaes":
        result = optimize_latent_cmaes(
            objective=objective,
            latent_dim=int(latent_dim),
            config=CMAESConfig(
                iterations=args.iterations,
                population_size=args.population_size,
                initial_sigma=args.initial_sigma,
                seed=args.seed,
                use_external_backend=(args.cma_backend == "auto"),
            ),
        )
    else:
        pop = args.population_size if args.population_size is not None else 96
        result = optimize_latent_cem(
            objective=objective,
            latent_dim=int(latent_dim),
            config=CEMConfig(
                iterations=args.iterations,
                population_size=pop,
                elite_fraction=args.elite_fraction,
                initial_std=args.initial_std,
                smoothing=args.smoothing,
                min_std=args.min_std,
                seed=args.seed,
            ),
        )

    if not best_genotype:
        best_genotype = _decode_from_z(model_name, model, result.best_z, device, idx2char)

    print(f"Model       : {model_name}")
    print(f"Algorithm   : {args.algorithm}")
    print(f"Iterations  : {args.iterations}")
    print(f"Best fitness: {result.best_score:.6f}")
    print(f"Best genotype: {best_genotype}")
    print(f"Best z      : {np.array2string(result.best_z, precision=4, separator=', ')}")


if __name__ == "__main__":
    main()
