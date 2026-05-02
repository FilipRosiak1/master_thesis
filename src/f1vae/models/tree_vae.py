from __future__ import annotations

import nltk
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from f1vae.grammars import f1 as G


class TreeAwareEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        pad_rule_idx: int,
        lhs_vocab_size: int,
    ) -> None:
        super().__init__()
        self.pad_rule_idx = pad_rule_idx
        self.rule_embedding = nn.Embedding(num_classes, emb_dim, padding_idx=pad_rule_idx)
        self.lhs_embedding = nn.Embedding(lhs_vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim * 2, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, lhs_context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rule_emb = self.rule_embedding(x)
        lhs_emb = self.lhs_embedding(lhs_context)
        features = torch.cat([rule_emb, lhs_emb], dim=-1)

        lengths = (x != self.pad_rule_idx).sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(features, lengths=lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class TreeAwareDecoder(nn.Module):
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

    def forward(
        self,
        z: torch.Tensor,
        target_seq: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        max_steps: int | None = None,
    ) -> torch.Tensor:
        del teacher_forcing_ratio
        steps = self.max_length if max_steps is None else min(self.max_length, max(1, int(max_steps)))
        batch_size = z.size(0)
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        input_rule = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        z_step = z.unsqueeze(1)

        if target_seq is None:
            step_logits: list[torch.Tensor] = []
            for _ in range(steps):
                embedded = self.embedding(input_rule)
                decoder_in = torch.cat([embedded, z_step], dim=-1)
                gru_out, hidden = self.gru(decoder_in, hidden)
                prediction = self.fc_out(gru_out.squeeze(1))
                step_logits.append(prediction.unsqueeze(1))
                input_rule = prediction.argmax(1).unsqueeze(1)
                if torch.all(input_rule.squeeze(1) == self.pad_rule_idx):
                    break
            return torch.cat(step_logits, dim=1)

        outputs = torch.zeros(batch_size, steps, self.num_classes, device=z.device)
        for t in range(steps):
            embedded = self.embedding(input_rule)
            decoder_in = torch.cat([embedded, z_step], dim=-1)
            gru_out, hidden = self.gru(decoder_in, hidden)
            prediction = self.fc_out(gru_out.squeeze(1))
            outputs[:, t, :] = prediction
            input_rule = target_seq[:, t].unsqueeze(1)

        return outputs


class TreeGrammarVAE(nn.Module):
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
        self.decoder = TreeAwareDecoder(
            num_classes=num_classes,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=pad_rule_idx,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
            stack = [self.start_symbol]
            context_row: list[int] = []
            for rule_idx in row:
                if rule_idx == self.pad_rule_idx:
                    context_row.append(self.default_lhs_idx)
                    continue

                next_lhs = stack.pop() if stack else "Nothing"
                context_row.append(self.lhs_map.get(next_lhs, self.default_lhs_idx))

                prod = self.productions[rule_idx]
                rhs_nts = [
                    str(sym)
                    for sym in prod.rhs()
                    if isinstance(sym, nltk.grammar.Nonterminal) and str(sym) != "None"
                ]
                stack.extend(rhs_nts[::-1])

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
