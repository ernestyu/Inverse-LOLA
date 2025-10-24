from __future__ import annotations

import torch
import torch.nn as nn


class JointRewardNetwork(nn.Module):
    """State-action reward approximator shared across agents."""

    def __init__(self, num_states: int, num_actions: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_embedding = nn.Embedding(num_states, hidden_dim)
        self.action_embedding_self = nn.Embedding(num_actions, hidden_dim)
        self.action_embedding_other = nn.Embedding(num_actions, hidden_dim)

        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_indices: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        state_feat = self.state_embedding(state_indices)
        action_self = self.action_embedding_self(joint_actions[..., 0])
        action_other = self.action_embedding_other(joint_actions[..., 1])
        features = torch.cat([state_feat, action_self, action_other], dim=-1)
        reward = self.trunk(features)
        return reward.squeeze(-1)

