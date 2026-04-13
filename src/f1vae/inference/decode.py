from __future__ import annotations

import nltk
import torch

from f1vae.grammars import f1 as G


def decode_char_indices(indices: torch.Tensor, idx2char: dict[int, str]) -> str:
    decoded = []
    for idx in indices.cpu().numpy():
        char = idx2char[int(idx)]
        if char in {"<EOS>", "<PAD>"}:
            break
        if char != "<SOS>":
            decoded.append(char)
    return "".join(decoded)


def decode_grammar_indices(indices: torch.Tensor) -> str:
    productions = G.GCFG.productions()
    stack = [G.GCFG.start()]
    result = []
    rule_ptr = 0

    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy()

    while stack and rule_ptr < len(indices):
        current = stack.pop()
        if isinstance(current, nltk.grammar.Nonterminal):
            prod = productions[int(indices[rule_ptr])]
            rule_ptr += 1

            if str(prod.lhs()) == "Nothing":
                continue

            for sym in reversed(prod.rhs()):
                if sym is not None:
                    stack.append(sym)
        else:
            result.append(str(current))

    return "".join(result)


def decode_masked_deterministic(model, z: torch.Tensor) -> torch.Tensor:
    batch_size = z.size(0)
    stacks = [[str(G.GCFG.start())] for _ in range(batch_size)]
    unmasked = model.decoder(z)
    sampled_indices = []

    productions = G.GCFG.productions()
    lhs_map = {str(lhs): i for i, lhs in enumerate(G.lhs_list)}

    for t in range(model.max_length):
        next_nonterminal = []
        for b_idx in range(batch_size):
            if stacks[b_idx]:
                next_nonterminal.append(lhs_map[stacks[b_idx].pop()])
            else:
                next_nonterminal.append(lhs_map.get("Nothing", 0))

        current_masks = model.masks[next_nonterminal]
        logits = unmasked[:, t, :]
        masked_probs = torch.exp(logits) * current_masks + 1e-10
        sampled_output = torch.argmax(masked_probs, dim=1)
        sampled_indices.append(sampled_output)

        for b_idx in range(batch_size):
            prod = productions[int(sampled_output[b_idx].item())]
            rhs_nts = [
                str(sym)
                for sym in prod.rhs()
                if isinstance(sym, nltk.grammar.Nonterminal) and str(sym) != "None"
            ]
            stacks[b_idx].extend(rhs_nts[::-1])

    return torch.stack(sampled_indices, dim=1)
