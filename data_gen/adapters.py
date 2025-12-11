"""Adapters connecting learners, envs, and data generators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(z)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.clip(denom, 1e-9, None)


@dataclass
class Trajectory:
    states: List[int]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]


@dataclass
class TabularPolicyModel:
    action_probs: np.ndarray  # shape: (num_states, num_actions)

    def probs(self, state: int) -> np.ndarray:
        return self.action_probs[state]


@dataclass
class AgentRollout:
    agent_id: str
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[float] | None = None


@dataclass
class PPORollout:
    agent_rollouts: Dict[str, AgentRollout]


@dataclass
class LearningPhaseData:
    phase_idx: int
    policy_params: Any
    policy_models: Any
    theta_sequence: Any
    trajectories: Any


def ma_spi_to_learning_phases(phases: Iterable[dict]) -> List[LearningPhaseData]:
    """Convert MA-SPI raw phase samples into LearningPhaseData list."""
    converted: List[LearningPhaseData] = []
    for raw in phases:
        logits = np.array(raw["policy_logits"], dtype=float)
        theta_sequence = [np.array(theta, dtype=float) for theta in raw.get("theta_sequence", [logits])]
        probs = _softmax(logits)
        policy_model = TabularPolicyModel(action_probs=probs)
        lp = LearningPhaseData(
            phase_idx=int(raw["phase_idx"]),
            policy_params=logits,
            policy_models=policy_model,
            theta_sequence=theta_sequence,
            trajectories=list(raw["trajectories"]),
        )
        converted.append(lp)
    return converted


def ppo_to_learning_phases(checkpoints: Iterable[dict]) -> List[LearningPhaseData]:
    """Convert PPO checkpoints (with rollouts) to LearningPhaseData list."""
    phases: List[LearningPhaseData] = []
    for idx, ckpt in enumerate(checkpoints):
        theta_vec = np.array(ckpt["theta"], dtype=float)
        rollouts: list[PPORollout] = ckpt["rollouts"]
        # Assume shared policy; map each agent to same theta vector.
        agent_ids = list(rollouts[0].agent_rollouts.keys()) if rollouts else []
        policy_params = {aid: theta_vec for aid in agent_ids}
        theta_sequence = {aid: [theta_vec] for aid in agent_ids}
        policy_models = {"shared_policy": "SharedPPO"}
        phases.append(
            LearningPhaseData(
                phase_idx=idx,
                policy_params=policy_params,
                policy_models=policy_models,
                theta_sequence=theta_sequence,
                trajectories=rollouts,
            )
        )
    return phases


__all__ = [
    "ma_spi_to_learning_phases",
    "ppo_to_learning_phases",
    "Trajectory",
    "AgentRollout",
    "PPORollout",
    "LearningPhaseData",
    "TabularPolicyModel",
]
