from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """State-conditional categorical policy with learned logits table."""

    def __init__(self, num_states: int, num_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.embedding = nn.Embedding(num_states, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state_indices: torch.Tensor) -> torch.Tensor:
        features = self.embedding(state_indices)
        logits = self.trunk(features)
        return logits

    def distribution(
        self, state_indices: torch.Tensor, temperature: float = 1.0
    ) -> Categorical:
        logits = self.forward(state_indices)
        if temperature <= 0:
            raise ValueError("Temperature must be strictly positive.")
        scaled_logits = logits / temperature
        return Categorical(logits=scaled_logits)

    def log_prob(
        self, state_indices: torch.Tensor, actions: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        dist = self.distribution(state_indices, temperature=temperature)
        return dist.log_prob(actions)

