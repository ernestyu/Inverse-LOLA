"""Reward network implementations."""
from __future__ import annotations

import torch
from torch import nn

from models.feature_maps import rbf_features


class LinearReward(nn.Module):
    """Linear reward: R = w^T phi."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features shape: (..., feature_dim)
        return torch.tensordot(features, self.weight, dims=([-1], [0]))


class RBFReward(nn.Module):
    """RBF reward with linear head over RBF features."""

    def __init__(self, centers: torch.Tensor, bandwidth: float):
        super().__init__()
        self.register_buffer("centers", centers)
        self.bandwidth = float(bandwidth)
        self.weight = nn.Parameter(torch.zeros(centers.shape[0]))

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        # sample shape: (..., feat_dim)
        phi = rbf_features(sample, self.centers, self.bandwidth)
        return torch.tensordot(phi, self.weight, dims=([-1], [0]))


__all__ = ["LinearReward", "RBFReward"]
