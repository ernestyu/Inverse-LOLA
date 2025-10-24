from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np
import torch

JointAction = Tuple[int, int]


@dataclass
class Trajectory:
    """On-policy trajectory segment produced under a fixed joint policy."""

    states: List[int] = field(default_factory=list)
    joint_actions: List[JointAction] = field(default_factory=list)
    next_states: List[int] = field(default_factory=list)
    rewards_agent_0: List[float] = field(default_factory=list)
    rewards_agent_1: List[float] = field(default_factory=list)
    action_log_probs_agent_0: List[float] = field(default_factory=list)
    action_log_probs_agent_1: List[float] = field(default_factory=list)

    def append(
        self,
        state_index: int,
        joint_action: JointAction,
        next_state_index: int,
        reward_agent_0: float,
        reward_agent_1: float,
        log_prob_agent_0: float,
        log_prob_agent_1: float,
    ) -> None:
        self.states.append(state_index)
        self.joint_actions.append(joint_action)
        self.next_states.append(next_state_index)
        self.rewards_agent_0.append(float(reward_agent_0))
        self.rewards_agent_1.append(float(reward_agent_1))
        self.action_log_probs_agent_0.append(float(log_prob_agent_0))
        self.action_log_probs_agent_1.append(float(log_prob_agent_1))

    def to_tensors(self) -> "TrajectoryTensors":
        """Return CPU tensors representing the trajectory data."""
        cpu_device = torch.device("cpu")
        return TrajectoryTensors(
            states=torch.tensor(self.states, dtype=torch.long, device=cpu_device),
            joint_actions=torch.tensor(self.joint_actions, dtype=torch.long, device=cpu_device),
            next_states=torch.tensor(self.next_states, dtype=torch.long, device=cpu_device),
            rewards_agent_0=torch.tensor(self.rewards_agent_0, dtype=torch.float32, device=cpu_device),
            rewards_agent_1=torch.tensor(self.rewards_agent_1, dtype=torch.float32, device=cpu_device),
            log_probs_agent_0=torch.tensor(self.action_log_probs_agent_0, dtype=torch.float32, device=cpu_device),
            log_probs_agent_1=torch.tensor(self.action_log_probs_agent_1, dtype=torch.float32, device=cpu_device),
        )


@dataclass
class TrajectoryTensors:
    states: torch.Tensor
    joint_actions: torch.Tensor
    next_states: torch.Tensor
    rewards_agent_0: torch.Tensor
    rewards_agent_1: torch.Tensor
    log_probs_agent_0: torch.Tensor
    log_probs_agent_1: torch.Tensor


@dataclass
class StageDataset:
    """Collection of trajectories generated at a fixed phase h."""

    trajectories: List[Trajectory] = field(default_factory=list)

    def add(self, trajectory: Trajectory) -> None:
        self.trajectories.append(trajectory)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.trajectories)

    def concatenated(self, device: torch.device, pin_memory: bool = False) -> TrajectoryTensors:
        if not self.trajectories:
            raise ValueError("StageDataset is empty.")

        tensors = [traj.to_tensors() for traj in self.trajectories]
        states = torch.cat([t.states for t in tensors], dim=0)
        joint_actions = torch.cat([t.joint_actions for t in tensors], dim=0)
        rewards_agent_0 = torch.cat([t.rewards_agent_0 for t in tensors], dim=0)
        rewards_agent_1 = torch.cat([t.rewards_agent_1 for t in tensors], dim=0)
        log_probs_agent_0 = torch.cat([t.log_probs_agent_0 for t in tensors], dim=0)
        log_probs_agent_1 = torch.cat([t.log_probs_agent_1 for t in tensors], dim=0)
        next_states = torch.cat([t.next_states for t in tensors], dim=0)

        if pin_memory:
            states = states.pin_memory()
            joint_actions = joint_actions.pin_memory()
            rewards_agent_0 = rewards_agent_0.pin_memory()
            rewards_agent_1 = rewards_agent_1.pin_memory()
            log_probs_agent_0 = log_probs_agent_0.pin_memory()
            log_probs_agent_1 = log_probs_agent_1.pin_memory()
            next_states = next_states.pin_memory()

        if device.type != "cpu":
            states = states.to(device, non_blocking=True)
            joint_actions = joint_actions.to(device, non_blocking=True)
            rewards_agent_0 = rewards_agent_0.to(device, non_blocking=True)
            rewards_agent_1 = rewards_agent_1.to(device, non_blocking=True)
            log_probs_agent_0 = log_probs_agent_0.to(device, non_blocking=True)
            log_probs_agent_1 = log_probs_agent_1.to(device, non_blocking=True)
            next_states = next_states.to(device, non_blocking=True)

        return TrajectoryTensors(
            states=states,
            joint_actions=joint_actions,
            next_states=next_states,
            rewards_agent_0=rewards_agent_0,
            rewards_agent_1=rewards_agent_1,
            log_probs_agent_0=log_probs_agent_0,
            log_probs_agent_1=log_probs_agent_1,
        )
