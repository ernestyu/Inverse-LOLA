"""MA-SPI sampling pipeline on the GridWorld environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from data_gen.adapters import Trajectory
from envs.gridworld import GridWorld
from learners.ma_spi.gridworld_ma_spi import MASPIGridWorld


@dataclass
class PhaseSample:
    phase_idx: int
    policy_logits: np.ndarray
    theta_sequence: List[np.ndarray]
    trajectories: List[Trajectory]


def run_ma_spi_phases(config: dict, seed: int = 0) -> List[PhaseSample]:
    num_phases = int(config.get("num_phases", 6))
    episodes_per_phase = int(config.get("episodes_per_phase", 5))
    max_steps = int(config.get("max_steps", 20))
    gamma = float(config.get("gamma", 0.95))
    lr = float(config.get("lr", 0.2))
    temperature = float(config.get("temperature", 1.0))

    env = GridWorld(max_steps=max_steps)
    learner = MASPIGridWorld(env, gamma=gamma, lr=lr, temperature=temperature, seed=seed)

    phases: List[PhaseSample] = []
    for phase_idx in range(num_phases):
        phase_trajs: List[Trajectory] = []
        for _ in range(episodes_per_phase):
            states, actions, rewards, dones = learner.generate_trajectory(max_steps=max_steps)
            phase_trajs.append(Trajectory(states=states, actions=actions, rewards=rewards, dones=dones))
        phase = PhaseSample(
            phase_idx=phase_idx,
            policy_logits=learner.policy.logits.copy(),
            theta_sequence=[learner.policy.logits.copy()],
            trajectories=phase_trajs,
        )
        phases.append(phase)
        learner.update_policy()
    return phases


__all__ = ["run_ma_spi_phases", "PhaseSample"]
