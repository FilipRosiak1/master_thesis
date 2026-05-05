from __future__ import annotations

import torch
import torch.nn as nn

from f1vae.grammars import f1 as G
from f1vae.models.tree_vae import TreeAwareEncoder


class MaskedTreeAwareDecoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_length: int,
        pad_rule_idx: int,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.num_classes = num_classes
        self.pad_rule_idx = pad_rule_idx
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(num_classes, emb_dim, padding_idx=pad_rule_idx)
        self.gru = nn.GRU(emb_dim + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

        self.register_buffer("masks", torch.tensor(G.masks, dtype=torch.bool))
        self.rhs_map = [list(reversed(rhs)) for rhs in G.rhs_map]
        self.start_lhs_idx = {str(lhs): i for i, lhs in enumerate(G.lhs_list)}[str(G.GCFG.start())]

    def _valid_mask_from_stacks(self, stacks: list[list[int]], device: torch.device) -> torch.Tensor:
        valid = torch.zeros(len(stacks), self.num_classes, dtype=torch.bool, device=device)
        active_rows: list[int] = []
        active_lhs: list[int] = []

        for row, stack in enumerate(stacks):
            if stack:
                active_rows.append(row)
                active_lhs.append(stack[-1])
            else:
                valid[row, self.pad_rule_idx] = True

        if active_rows:
            valid[active_rows, : self.masks.size(1)] = self.masks[active_lhs]

        return valid

    def _advance_stacks(self, stacks: list[list[int]], selected_rules: torch.Tensor) -> None:
        for row, rule_idx in enumerate(selected_rules.detach().cpu().tolist()):
            if not stacks[row] or rule_idx == self.pad_rule_idx:
                continue
            stacks[row].pop()
            if 0 <= rule_idx < len(self.rhs_map):
                stacks[row].extend(self.rhs_map[rule_idx])

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
            embedded = self.embedding(input_rule)
            decoder_in = torch.cat([embedded, z_step], dim=-1)
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


class MaskedTreeGrammarVAE(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_length: int,
        pad_rule_idx: int,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_rule_idx = pad_rule_idx
        self.productions = G.GCFG.productions()
        self.lhs_map = {str(lhs): i for i, lhs in enumerate(G.lhs_list)}
        self.start_symbol = str(G.GCFG.start())
        self.default_lhs_idx = self.lhs_map.get("Nothing", 0)

        self.encoder = TreeAwareEncoder(
            num_classes=num_classes,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            pad_rule_idx=pad_rule_idx,
            lhs_vocab_size=len(self.lhs_map),
        )
        self.decoder = MaskedTreeAwareDecoder(
            num_classes=num_classes,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=pad_rule_idx,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lhs_context = self._lhs_context_indices(x)
        return self.encoder(x, lhs_context)

    def decode(
        self,
        z: torch.Tensor,
        target_seq: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        max_steps: int | None = None,
    ) -> torch.Tensor:
        return self.decoder(z, target_seq, teacher_forcing_ratio, max_steps)

    def _lhs_context_indices(self, x: torch.Tensor) -> torch.Tensor:
        batch_context: list[list[int]] = []
        for row in x.detach().cpu().tolist():
            stack = [self.lhs_map[self.start_symbol]]
            context_row: list[int] = []
            for rule_idx in row:
                if rule_idx == self.pad_rule_idx:
                    context_row.append(self.default_lhs_idx)
                    continue

                next_lhs = stack.pop() if stack else self.default_lhs_idx
                context_row.append(next_lhs)
                if 0 <= rule_idx < len(G.rhs_map):
                    stack.extend(reversed(G.rhs_map[rule_idx]))

            batch_context.append(context_row)

        return torch.tensor(batch_context, dtype=torch.long, device=x.device)

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, x, teacher_forcing_ratio)
        return reconstruction, mu, logvar
