from __future__ import annotations

import torch
import torch.nn as nn


class PotentialNetwork(nn.Module):
    """State-dependent potential g_h(s) used for reward shaping."""

    def __init__(self, num_states: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_states, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        features = self.embedding(state_indices)
        potential = self.trunk(features)
        return potential.squeeze(-1)

