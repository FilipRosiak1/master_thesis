from __future__ import annotations

import torch
import torch.nn as nn


class EncoderSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embedding(x)
        _, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)
        return self.fc_mu(hidden), self.fc_logvar(hidden)


class DecoderSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, latent_dim: int, max_length: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        z: torch.Tensor,
        target_seq: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = z.size(0)
        hidden = self.latent_to_hidden(z).unsqueeze(0)
        input_token = torch.ones(batch_size, 1, dtype=torch.long, device=z.device)
        outputs = torch.zeros(batch_size, self.max_length, self.vocab_size, device=z.device)

        for t in range(self.max_length):
            embedded = self.embedding(input_token)
            gru_out, hidden = self.gru(embedded, hidden)
            prediction = self.fc_out(gru_out.squeeze(1))
            outputs[:, t, :] = prediction

            if target_seq is not None and torch.rand(1).item() < teacher_forcing_ratio:
                input_token = target_seq[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(1).unsqueeze(1)
        return outputs


class CharVAE(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, latent_dim: int, max_length: int) -> None:
        super().__init__()
        self.encoder = EncoderSeq2Seq(vocab_size, emb_dim, hidden_dim, latent_dim)
        self.decoder = DecoderSeq2Seq(vocab_size, emb_dim, hidden_dim, latent_dim, max_length)

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

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.decoder(z, None, teacher_forcing_ratio=0.0)
            return logits.argmax(-1)
