from __future__ import annotations

import torch
import torch.nn as nn

from f1vae.grammars import f1 as G
from f1vae.models.tree_vae_masked import MaskedTreeAwareDecoder, MaskedTreeGrammarVAE


class LHSConditionedMaskedTreeAwareDecoder(MaskedTreeAwareDecoder):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_length: int,
        pad_rule_idx: int,
        lhs_vocab_size: int,
    ) -> None:
        super().__init__(num_classes, emb_dim, hidden_dim, latent_dim, max_length, pad_rule_idx)
        self.lhs_embedding = nn.Embedding(lhs_vocab_size, emb_dim)
        self.gru = nn.GRU((emb_dim * 2) + latent_dim, hidden_dim, batch_first=True)
        self.finished_lhs_idx = {str(lhs): i for i, lhs in enumerate(G.lhs_list)}.get("Nothing", 0)

    def _lhs_input_from_stacks(self, stacks: list[list[int]], device: torch.device) -> torch.Tensor:
        lhs_indices = [stack[-1] if stack else self.finished_lhs_idx for stack in stacks]
        return torch.tensor(lhs_indices, dtype=torch.long, device=device).unsqueeze(1)

    def _decoder_input(self, input_rule: torch.Tensor, stacks: list[list[int]], z_step: torch.Tensor) -> torch.Tensor:
        rule_embedded = self.embedding(input_rule)
        lhs_embedded = self.lhs_embedding(self._lhs_input_from_stacks(stacks, z_step.device))
        return torch.cat([rule_embedded, lhs_embedded, z_step], dim=-1)

    def forward(
        self,
        z: torch.Tensor,
        target_seq: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        max_steps: int | None = None,
    ) -> torch.Tensor:
        steps = self.max_length if max_steps is None else min(self.max_length, max(1, int(max_steps)))
        batch_size = z.size(0)
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        input_rule = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        z_step = z.unsqueeze(1)
        stacks = [[self.start_lhs_idx] for _ in range(batch_size)]
        step_logits: list[torch.Tensor] = []

        for t in range(steps):
            decoder_in = self._decoder_input(input_rule, stacks, z_step)
            gru_out, hidden = self.gru(decoder_in, hidden)
            logits = self.fc_out(gru_out.squeeze(1))
            valid_mask = self._valid_mask_from_stacks(stacks, z.device)
            masked_logits = logits.masked_fill(~valid_mask, -1e9)
            step_logits.append(masked_logits.unsqueeze(1))

            predicted_rule = masked_logits.argmax(1)
            if target_seq is not None:
                target_rule = target_seq[:, t]
                if torch.rand(1).item() < teacher_forcing_ratio:
                    input_rule = target_rule.unsqueeze(1)
                else:
                    input_rule = predicted_rule.unsqueeze(1)
                self._advance_stacks(stacks, target_rule)
            else:
                input_rule = predicted_rule.unsqueeze(1)
                self._advance_stacks(stacks, predicted_rule)
                if torch.all(predicted_rule == self.pad_rule_idx):
                    break

        return torch.cat(step_logits, dim=1)


class LHSDepthConditionedMaskedTreeAwareDecoder(LHSConditionedMaskedTreeAwareDecoder):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_length: int,
        pad_rule_idx: int,
        lhs_vocab_size: int,
    ) -> None:
        super().__init__(num_classes, emb_dim, hidden_dim, latent_dim, max_length, pad_rule_idx, lhs_vocab_size)
        self.max_depth = max_length + 1
        self.depth_embedding = nn.Embedding(self.max_depth, emb_dim)
        self.gru = nn.GRU((emb_dim * 3) + latent_dim, hidden_dim, batch_first=True)

    def _depth_input_from_stacks(self, stacks: list[list[int]], device: torch.device) -> torch.Tensor:
        depth_indices = [min(len(stack), self.max_depth - 1) for stack in stacks]
        return torch.tensor(depth_indices, dtype=torch.long, device=device).unsqueeze(1)

    def _decoder_input(self, input_rule: torch.Tensor, stacks: list[list[int]], z_step: torch.Tensor) -> torch.Tensor:
        rule_embedded = self.embedding(input_rule)
        lhs_embedded = self.lhs_embedding(self._lhs_input_from_stacks(stacks, z_step.device))
        depth_embedded = self.depth_embedding(self._depth_input_from_stacks(stacks, z_step.device))
        return torch.cat([rule_embedded, lhs_embedded, depth_embedded, z_step], dim=-1)


class LHSConditionedMaskedTreeGrammarVAE(MaskedTreeGrammarVAE):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_length: int,
        pad_rule_idx: int,
    ) -> None:
        super().__init__(num_classes, emb_dim, hidden_dim, latent_dim, max_length, pad_rule_idx)
        self.decoder = LHSConditionedMaskedTreeAwareDecoder(
            num_classes=num_classes,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=pad_rule_idx,
            lhs_vocab_size=len(self.lhs_map),
        )


class LHSDepthConditionedMaskedTreeGrammarVAE(MaskedTreeGrammarVAE):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_length: int,
        pad_rule_idx: int,
    ) -> None:
        super().__init__(num_classes, emb_dim, hidden_dim, latent_dim, max_length, pad_rule_idx)
        self.decoder = LHSDepthConditionedMaskedTreeAwareDecoder(
            num_classes=num_classes,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=pad_rule_idx,
            lhs_vocab_size=len(self.lhs_map),
        )
