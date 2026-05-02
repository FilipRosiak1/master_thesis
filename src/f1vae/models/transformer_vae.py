from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_length, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.embedding(positions)


class TransformerRuleEncoder(nn.Module):
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
        self.pad_rule_idx = pad_rule_idx
        self.embedding = nn.Embedding(num_classes, emb_dim, padding_idx=pad_rule_idx)
        self.position = PositionalEncoding(emb_dim, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=max(1, emb_dim // 16),
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.fc_mu = nn.Linear(emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(emb_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding_mask = x == self.pad_rule_idx
        hidden = self.position(self.embedding(x))
        encoded = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        valid = (~key_padding_mask).unsqueeze(-1)
        denom = valid.sum(dim=1).clamp(min=1)
        pooled = (encoded * valid).sum(dim=1) / denom
        return self.fc_mu(pooled), self.fc_logvar(pooled)


class TransformerRuleDecoder(nn.Module):
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
        self.embedding = nn.Embedding(num_classes, emb_dim, padding_idx=pad_rule_idx)
        self.position = PositionalEncoding(emb_dim, max_length)
        self.latent_to_memory = nn.Linear(latent_dim, emb_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=emb_dim,
            nhead=max(1, emb_dim // 16),
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=2)
        self.fc_out = nn.Linear(emb_dim, num_classes)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def _decode_step(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        tgt = self.position(self.embedding(tokens))
        tgt_mask = self._causal_mask(tokens.size(1), tokens.device)
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.fc_out(decoded)

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
        memory = self.latent_to_memory(z).unsqueeze(1)

        if target_seq is not None:
            start = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
            decoder_input = torch.cat([start, target_seq[:, : steps - 1]], dim=1)
            return self._decode_step(memory, decoder_input)

        tokens = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
        step_logits: list[torch.Tensor] = []
        for _ in range(steps):
            logits = self._decode_step(memory, tokens)
            next_logits = logits[:, -1:, :]
            step_logits.append(next_logits)
            next_token = next_logits.argmax(dim=-1)
            tokens = torch.cat([tokens, next_token], dim=1)
            if torch.all(next_token.squeeze(1) == self.pad_rule_idx):
                break

        return torch.cat(step_logits, dim=1)


class TransformerGrammarVAE(nn.Module):
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
        self.encoder = TransformerRuleEncoder(
            num_classes=num_classes,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_length=max_length,
            pad_rule_idx=pad_rule_idx,
        )
        self.decoder = TransformerRuleDecoder(
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

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z, x, teacher_forcing_ratio)
        return reconstruction, mu, logvar
