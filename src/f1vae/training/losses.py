from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sequence_vae_loss(recon_logits: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, pad_idx: int, beta: float = 1.0):
    recon_flat = recon_logits.view(-1, recon_logits.size(-1))
    target_flat = target.view(-1)
    ce = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="sum")(recon_flat, target_flat)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return ce + beta * kld, ce, kld


def grammar_masked_vae_loss(
    recon_logits: torch.Tensor,
    target_one_hot: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    masks: torch.Tensor,
    ind_of_ind: torch.Tensor,
    beta: float = 1.0,
):
    _, _, n_chars = target_one_hot.size()

    rule_indices = torch.argmax(target_one_hot, dim=-1)
    lhs_indices = ind_of_ind[rule_indices]
    current_masks = masks[lhs_indices]

    masked_logits = recon_logits.clone()
    masked_logits[current_masks == 0] = -1e10

    recon_flat = masked_logits.view(-1, n_chars)
    target_flat = rule_indices.view(-1)

    ce = F.cross_entropy(recon_flat, target_flat, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return ce + beta * kld, ce, kld
