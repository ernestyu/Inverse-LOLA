"""Feature mapping utilities for GridWorld, MPE, and MultiWalker."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import torch

from data_gen.adapters import PPORollout


def indicator_feature_gridworld(
    state: int | torch.Tensor,
    action: int | torch.Tensor,
    n_states: int = 9,
    n_actions: int = 5,
    concat: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """One-hot over states and actions for GridWorld."""
    s = torch.as_tensor(state, device=device, dtype=torch.long)
    a = torch.as_tensor(action, device=device, dtype=torch.long)
    s_oh = torch.nn.functional.one_hot(s, num_classes=n_states).float()
    a_oh = torch.nn.functional.one_hot(a, num_classes=n_actions).float()
    return torch.cat([s_oh, a_oh], dim=-1) if concat else (s_oh, a_oh)


def _collect_rollout_samples(
    rollouts: Iterable[PPORollout],
    use_actions: bool = True,
    max_samples: int = 5000,
) -> np.ndarray:
    samples: List[np.ndarray] = []
    for ro in rollouts:
        for agent_rollout in ro.agent_rollouts.values():
            obs_seq = agent_rollout.observations
            act_seq = agent_rollout.actions if use_actions else [None] * len(obs_seq)
            for obs, act in zip(obs_seq, act_seq):
                obs_vec = np.asarray(obs, dtype=np.float32).reshape(-1)
                if use_actions and act is not None:
                    act_vec = np.asarray(act, dtype=np.float32).reshape(-1)
                    sample = np.concatenate([obs_vec, act_vec], axis=0)
                else:
                    sample = obs_vec
                samples.append(sample)
                if len(samples) >= max_samples:
                    return np.stack(samples, axis=0)
    if not samples:
        raise ValueError("No samples found in rollouts to build RBF centers.")
    return np.stack(samples, axis=0)


def estimate_rbf_centers_from_rollouts(
    rollouts: Iterable[PPORollout],
    num_centers: int = 16,
    use_actions: bool = True,
    max_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, float]:
    """Estimate RBF centers/bandwidth from PPO rollouts."""
    rng = rng or np.random.default_rng()
    samples = _collect_rollout_samples(rollouts, use_actions=use_actions, max_samples=max_samples)
    n = samples.shape[0]
    if n <= num_centers:
        centers = samples
    else:
        idx = rng.choice(n, size=num_centers, replace=False)
        centers = samples[idx]
    std = samples.std(axis=0).mean()
    bandwidth = float(max(std, 1e-2))
    return centers.astype(np.float32), bandwidth


def rbf_features(
    sample: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Compute RBF features phi_i = exp(-||x-c_i||^2 / (2*sigma^2))."""
    sample = sample.unsqueeze(0) if sample.ndim == 1 else sample
    diff = sample[:, None, :] - centers[None, :, :]
    dist2 = (diff ** 2).sum(dim=-1)
    phi = torch.exp(-dist2 / (2 * (bandwidth ** 2)))
    return phi if sample.ndim == 2 else phi.squeeze(0)


def mpe_rbf_features(
    obs: np.ndarray | torch.Tensor,
    action: np.ndarray | torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    sample = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
    act = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
    vec = torch.cat([sample, act], dim=0)
    return rbf_features(vec, centers, bandwidth)


def multiwalker_rbf_features(
    obs: np.ndarray | torch.Tensor,
    action: np.ndarray | torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    sample = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
    act = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
    vec = torch.cat([sample, act], dim=0)
    return rbf_features(vec, centers, bandwidth)


def mpe_simple_features(
    obs: np.ndarray | torch.Tensor,
    action: np.ndarray | torch.Tensor,
    scale: float = 5.0,
    action_dim: int | None = None,
) -> torch.Tensor:
    """Simple non-zero feature: scaled observation + action one-hot (argmax)."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32).reshape(-1) / scale
    act_t = torch.as_tensor(action, dtype=torch.float32).reshape(-1)
    action_dim = action_dim or act_t.numel()
    idx = int(torch.argmax(act_t).item()) % action_dim
    one_hot = torch.zeros(action_dim, dtype=torch.float32, device=obs_t.device)
    one_hot[idx] = 1.0
    return torch.cat([obs_t, one_hot], dim=0)


__all__ = [
    "indicator_feature_gridworld",
    "estimate_rbf_centers_from_rollouts",
    "rbf_features",
    "mpe_rbf_features",
    "multiwalker_rbf_features",
    "mpe_simple_features",
]
