from __future__ import annotations

import torch
import torch.nn as nn


class GrammarRuleEncoder(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, hidden_dim: int, latent_dim: int, pad_rule_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, emb_dim, padding_idx=pad_rule_idx)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class GrammarRuleDecoder(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int, hidden_dim: int, latent_dim: int, max_length: int, pad_rule_idx: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.num_classes = num_classes
        self.pad_rule_idx = pad_rule_idx
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.embedding = nn.Embedding(num_classes, emb_dim, padding_idx=pad_rule_idx)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

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

        if target_seq is None:
            step_logits: list[torch.Tensor] = []
            for _ in range(steps):
                embedded = self.embedding(input_rule)
                gru_out, hidden = self.gru(embedded, hidden)
                prediction = self.fc_out(gru_out.squeeze(1))
                step_logits.append(prediction.unsqueeze(1))
                input_rule = prediction.argmax(1).unsqueeze(1)
                if torch.all(input_rule.squeeze(1) == self.pad_rule_idx):
                    break
            return torch.cat(step_logits, dim=1)

        outputs = torch.zeros(batch_size, steps, self.num_classes, device=z.device)
        for t in range(steps):
            embedded = self.embedding(input_rule)
            gru_out, hidden = self.gru(embedded, hidden)
            prediction = self.fc_out(gru_out.squeeze(1))
            outputs[:, t, :] = prediction
            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_rule = target_seq[:, t].unsqueeze(1)
            else:
                input_rule = prediction.argmax(1).unsqueeze(1)
        return outputs


class GrammarRuleVAE(nn.Module):
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
        self.encoder = GrammarRuleEncoder(num_classes, emb_dim, hidden_dim, latent_dim, pad_rule_idx)
        self.decoder = GrammarRuleDecoder(num_classes, emb_dim, hidden_dim, latent_dim, max_length, pad_rule_idx)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z, x, teacher_forcing_ratio)
        return reconstruction, mu, logvar
