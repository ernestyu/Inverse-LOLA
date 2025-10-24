from __future__ import annotations

import torch
import torch.nn as nn


class SoftQNetwork(nn.Module):
    """Soft Q-network for a single agent."""

    def __init__(self, num_states: int, num_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions

        self.state_embedding = nn.Embedding(num_states, hidden_dim)
        self.action_embedding = nn.Embedding(num_actions, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_indices: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_features = self.state_embedding(state_indices)
        action_features = self.action_embedding(actions)
        values = self.trunk(torch.cat([state_features, action_features], dim=-1))
        return values.squeeze(-1)

    def all_action_values(self, state_indices: torch.Tensor) -> torch.Tensor:
        state_features = self.state_embedding(state_indices)
        expanded_state = state_features.unsqueeze(1).repeat(1, self.num_actions, 1)
        action_indices = torch.arange(self.num_actions, device=state_indices.device)
        action_features = self.action_embedding(action_indices)
        action_features = action_features.unsqueeze(0).expand(expanded_state.shape[0], -1, -1)
        logits = self.trunk(torch.cat([expanded_state, action_features], dim=-1))
        return logits.squeeze(-1)

