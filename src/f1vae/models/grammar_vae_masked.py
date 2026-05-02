from __future__ import annotations

import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F

from f1vae.grammars import f1 as G


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, max_length: int, n_chars: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_chars, 9, kernel_size=9)
        self.conv2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv3 = nn.Conv1d(9, 10, kernel_size=11)

        l1 = max_length - 8
        l2 = l1 - 8
        l3 = l2 - 10
        self.flatten_size = 10 * l3

        self.fc1 = nn.Linear(self.flatten_size, 435)
        self.fc_mu = nn.Linear(435, latent_dim)
        self.fc_logvar = nn.Linear(435, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, max_length: int, n_chars: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.gru1 = nn.GRU(latent_dim, 501, batch_first=True)
        self.gru2 = nn.GRU(501, 501, batch_first=True)
        self.gru3 = nn.GRU(501, 501, batch_first=True)
        self.fc_out = nn.Linear(501, n_chars)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        h = h.unsqueeze(1).repeat(1, self.max_length, 1)
        h, _ = self.gru1(h)
        h, _ = self.gru2(h)
        h, _ = self.gru3(h)
        return self.fc_out(h)


class GrammarMaskedVAE(nn.Module):
    def __init__(self, latent_dim: int, max_length: int, n_chars: int) -> None:
        super().__init__()
        self.max_length = max_length
        self.encoder = Encoder(latent_dim, max_length, n_chars)
        self.decoder = Decoder(latent_dim, max_length, n_chars)
        self.register_buffer("masks", torch.tensor(G.masks, dtype=torch.float32))
        self.register_buffer("ind_of_ind", torch.tensor(G.ind_of_ind, dtype=torch.long))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode_masked(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        stacks = [[str(G.GCFG.start())] for _ in range(batch_size)]
        unmasked = self.decoder(z)
        x_hat = torch.zeros_like(unmasked)

        productions = G.GCFG.productions()
        lhs_map = {str(lhs): i for i, lhs in enumerate(G.lhs_list)}

        for t in range(self.max_length):
            next_nonterminal = []
            for b_ix in range(batch_size):
                if stacks[b_ix]:
                    next_nonterminal.append(lhs_map[stacks[b_ix].pop()])
                else:
                    next_nonterminal.append(lhs_map.get("Nothing", 0))

            current_masks = self.masks[next_nonterminal]
            logits = unmasked[:, t, :]
            masked_probs = torch.exp(logits) * current_masks + 1e-10
            sampled_output = torch.multinomial(masked_probs, 1).squeeze(1)
            x_hat[torch.arange(batch_size), t, sampled_output] = 1.0

            for b_ix in range(batch_size):
                prod = productions[sampled_output[b_ix].item()]
                rhs_nts = [
                    str(sym)
                    for sym in prod.rhs()
                    if isinstance(sym, nltk.grammar.Nonterminal) and str(sym) != "None"
                ]
                stacks[b_ix].extend(rhs_nts[::-1])

        return x_hat
