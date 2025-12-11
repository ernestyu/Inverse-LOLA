"""MA-LfL core on tabular GridWorld."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List

import numpy as np
import torch

from data_gen.adapters import LearningPhaseData, Trajectory, TabularPolicyModel
from models.feature_maps import indicator_feature_gridworld


@dataclass
class MALfLOutput:
    w_hat: np.ndarray
    feature_dim: int
    debug: dict


def rollout_expectations(trajectory: Trajectory, feature_fn: Callable[[int, int], torch.Tensor]) -> torch.Tensor:
    feats = []
    for s, a in zip(trajectory.states[:-1], trajectory.actions):
        feats.append(feature_fn(s, a))
    return torch.stack(feats, dim=0).sum(dim=0)


def ma_lfl_gridworld(phases: Iterable[LearningPhaseData], n_states: int = 9, n_actions: int = 5) -> MALfLOutput:
    """Recover reward weights for GridWorld using a simple moment-matching LfL."""
    feature_dim = n_states + n_actions
    phi = lambda s, a: indicator_feature_gridworld(s, a, n_states=n_states, n_actions=n_actions)  # noqa: E731

    # Aggregate feature expectations across phases/trajectories
    feat_totals = torch.zeros(feature_dim, dtype=torch.float32)
    count = 0
    for phase in phases:
        trajs: List[Trajectory] = phase.trajectories
        for traj in trajs:
            feat_totals += rollout_expectations(traj, phi)
            count += len(traj.actions)
    if count == 0:
        raise ValueError("No transitions found in phases.")
    empirical_feat = feat_totals / max(count, 1)

    # Simple L2-regularized least squares against policy logits (proxy for advantage)
    # Flatten policy logits as a pseudo target; this is a placeholder recovery heuristic.
    logits_stack = []
    for phase in phases:
        policy = phase.policy_models
        if isinstance(policy, TabularPolicyModel):
            logits_stack.append(torch.tensor(policy.action_probs, dtype=torch.float32))
        else:
            # fallback if stored differently
            logits_stack.append(torch.tensor(phase.policy_params, dtype=torch.float32))
    if not logits_stack:
        raise ValueError("No policy params/logits available for recovery.")
    pseudo_targets = torch.cat([x.flatten() for x in logits_stack], dim=0)
    target_mean = pseudo_targets.mean()

    # Solve w_hat via ridge regression on empirical features ~ target_mean
    lam = 1e-2
    A = torch.eye(feature_dim) * lam
    b = empirical_feat * target_mean
    w_hat = torch.linalg.solve(A + torch.eye(feature_dim), b)  # simple closed form

    return MALfLOutput(w_hat=w_hat.detach().numpy(), feature_dim=feature_dim, debug={"empirical_feat": empirical_feat})


__all__ = ["ma_lfl_gridworld", "MALfLOutput"]
