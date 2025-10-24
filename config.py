from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class EnvironmentConfig:
    grid_size: int
    start_positions: list[list[int]]
    goal_position: list[int]
    reward_family: str


@dataclass
class MASPIConfig:
    num_iterations: int
    evaluation_episodes_per_iteration: int
    episode_length: int
    q_updates_per_stage: int
    policy_updates_per_stage: int
    seed: int
    update_batch_size: int = 2048
    min_update_batch_size: int = 512
    batch_shrink_factor: float = 0.5
    batch_growth_factor: float = 1.2
    cache_stage_on_gpu: bool = False
    pin_memory: bool = True
    use_amp: bool = True
    num_workers: int = 0


@dataclass
class MALFLConfig:
    reward_epochs: int
    reward_lr: float
    shaping_lr: float
    potential_epochs: int
    kl_penalty: float
    prob_clip: float
    potential_reg_weight: float
    policy_estimation_lr: float
    policy_estimation_epochs: int
    policy_entropy_coef: float
    policy_estimation_batch: int
    reward_batch_size: int
    num_workers: int = 0


@dataclass
class OptimizationConfig:
    policy_lr: float
    q_lr: float
    value_regularizer: float


@dataclass
class LoggingConfig:
    base_dir: str
    save_trajectories: bool
    save_policies: bool
    save_models: bool


@dataclass
class ExperimentConfig:
    alpha: float
    gamma: float
    horizon: int
    device: str
    environment: EnvironmentConfig
    maspi: MASPIConfig
    malfl: MALFLConfig
    optimization: OptimizationConfig
    logging: LoggingConfig

    @property
    def num_states(self) -> int:
        size = self.environment.grid_size
        size_sq = size * size
        return size_sq * size_sq

    @property
    def num_actions(self) -> int:
        return 5


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    env = EnvironmentConfig(**raw["environment"])
    maspi = MASPIConfig(**raw["maspi"])
    malfl = MALFLConfig(**raw["malfl"])
    optim_cfg = OptimizationConfig(**raw["optimization"])
    logging = LoggingConfig(**raw["logging"])

    return ExperimentConfig(
        alpha=float(raw["alpha"]),
        gamma=float(raw["gamma"]),
        horizon=int(raw["horizon"]),
        device=raw.get("device", "cpu"),
        environment=env,
        maspi=maspi,
        malfl=malfl,
        optimization=optim_cfg,
        logging=logging,
    )
