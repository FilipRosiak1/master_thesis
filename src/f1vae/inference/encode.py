from __future__ import annotations

import nltk
import numpy as np
import torch

from f1vae.grammars import f1 as G

_PARSER = nltk.BottomUpChartParser(G.GCFG)
_PRODUCTIONS = G.GCFG.productions()
_RULE2IDX = {prod: i for i, prod in enumerate(_PRODUCTIONS)}


def encode_char_string(genotype: str, char2idx: dict[str, int], max_length: int) -> torch.Tensor:
    tokens = ["<SOS>"] + list(genotype) + ["<EOS>"]
    try:
        encoded = [char2idx[t] for t in tokens]
    except KeyError as exc:
        raise ValueError(f"Unknown character for this model vocabulary: {exc}") from exc

    pad_idx = char2idx["<PAD>"]
    if len(encoded) < max_length:
        encoded.extend([pad_idx] * (max_length - len(encoded)))
    else:
        encoded = encoded[:max_length]

    return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def _productions_for(genotype: str, max_length: int) -> list[int]:
    trees = list(_PARSER.parse(list(genotype)))
    if not trees:
        raise ValueError("Input cannot be parsed by grammar")
    seq = trees[0].productions()
    if len(seq) > max_length:
        raise ValueError(f"Input is too long for max_length={max_length}")
    return [_RULE2IDX[prod] for prod in seq]


def encode_grammar_rule_string(genotype: str, max_length: int) -> torch.Tensor:
    indices = _productions_for(genotype, max_length)
    pad_rule_idx = len(_PRODUCTIONS)
    indices.extend([pad_rule_idx] * (max_length - len(indices)))
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


def encode_grammar_onehot_string(genotype: str, max_length: int) -> torch.Tensor:
    indices = _productions_for(genotype, max_length)
    one_hot = np.zeros((max_length, len(_PRODUCTIONS)), dtype=np.float32)
    for t, idx in enumerate(indices):
        one_hot[t, idx] = 1.0
    if len(indices) < max_length:
        one_hot[len(indices) :, -1] = 1.0
    return torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
