from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQRuleEncoder(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, hidden_dim: int, latent_dim: int, pad_rule_idx: int) -> None:
        super().__init__()
        self.pad_rule_idx = pad_rule_idx
        self.embedding = nn.Embedding(num_classes, emb_dim, padding_idx=pad_rule_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)
        return self.fc_latent(hidden)


class VQRuleDecoder(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, hidden_dim: int, latent_dim: int, max_length: int, pad_rule_idx: int) -> None:
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


class VQGrammarAE(nn.Module):
    def __init__(
        self,
        num_classes: int,
        emb_dim: int,
        hidden_dim: int,
        latent_dim: int,
        max_length: int,
        pad_rule_idx: int,
        *,
        num_codes: int = 256,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.commitment_cost = commitment_cost
        self.encoder = VQRuleEncoder(num_classes, emb_dim, hidden_dim, latent_dim, pad_rule_idx)
        self.decoder = VQRuleDecoder(num_classes, emb_dim, hidden_dim, latent_dim, max_length, pad_rule_idx)
        self.codebook = nn.Embedding(num_codes, latent_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)
        self.last_vq_loss = torch.tensor(0.0)

    def _quantize(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1)
            - 2.0 * z_e @ self.codebook.weight.t()
        )
        code_indices = distances.argmin(dim=1)
        z_q = self.codebook(code_indices)
        return z_q, code_indices

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encoder(x)
        z_q, _ = self._quantize(z_e)
        z_st = z_e + (z_q - z_e).detach()

        self.last_vq_loss = F.mse_loss(z_q.detach(), z_e) + self.commitment_cost * F.mse_loss(z_q, z_e.detach())
        reconstruction = self.decoder(z_st, x, teacher_forcing_ratio)

        zero_logvar = torch.zeros_like(z_q)
        return reconstruction, z_q, zero_logvar

    def aux_loss(self) -> torch.Tensor:
        return self.last_vq_loss
