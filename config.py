from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


# -----------------------------
# Dataclasses for each section
# -----------------------------

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
        return size * size * size * size

    @property
    def num_actions(self) -> int:
        return 5


# -----------------------------
# Helper: safe numeric conversion
# -----------------------------

def _as_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        raise TypeError(f"Expected numeric value, got {x!r}")


def _as_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        raise TypeError(f"Expected integer value, got {x!r}")


# -----------------------------
# Config loader
# -----------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    """Load configuration YAML and safely cast numeric fields."""
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    # Environment
    env = EnvironmentConfig(**raw["environment"])

    # MASPI
    maspi_raw = raw["maspi"]
    maspi = MASPIConfig(
        num_iterations=_as_int(maspi_raw["num_iterations"]),
        evaluation_episodes_per_iteration=_as_int(maspi_raw["evaluation_episodes_per_iteration"]),
        episode_length=_as_int(maspi_raw["episode_length"]),
        q_updates_per_stage=_as_int(maspi_raw["q_updates_per_stage"]),
        policy_updates_per_stage=_as_int(maspi_raw["policy_updates_per_stage"]),
        seed=_as_int(maspi_raw["seed"]),
        update_batch_size=_as_int(maspi_raw.get("update_batch_size", 2048)),
        min_update_batch_size=_as_int(maspi_raw.get("min_update_batch_size", 512)),
        batch_shrink_factor=_as_float(maspi_raw.get("batch_shrink_factor", 0.5)),
        batch_growth_factor=_as_float(maspi_raw.get("batch_growth_factor", 1.2)),
        cache_stage_on_gpu=bool(maspi_raw.get("cache_stage_on_gpu", False)),
        pin_memory=bool(maspi_raw.get("pin_memory", True)),
        use_amp=bool(maspi_raw.get("use_amp", True)),
        num_workers=_as_int(maspi_raw.get("num_workers", 0)),
    )

    # MALfL
    malfl_raw = raw["malfl"]
    malfl = MALFLConfig(
        reward_epochs=_as_int(malfl_raw["reward_epochs"]),
        reward_lr=_as_float(malfl_raw["reward_lr"]),
        shaping_lr=_as_float(malfl_raw["shaping_lr"]),
        potential_epochs=_as_int(malfl_raw["potential_epochs"]),
        kl_penalty=_as_float(malfl_raw["kl_penalty"]),
        prob_clip=_as_float(malfl_raw["prob_clip"]),
        potential_reg_weight=_as_float(malfl_raw["potential_reg_weight"]),
        policy_estimation_lr=_as_float(malfl_raw["policy_estimation_lr"]),
        policy_estimation_epochs=_as_int(malfl_raw["policy_estimation_epochs"]),
        policy_entropy_coef=_as_float(malfl_raw["policy_entropy_coef"]),
        policy_estimation_batch=_as_int(malfl_raw["policy_estimation_batch"]),
        reward_batch_size=_as_int(malfl_raw["reward_batch_size"]),
        num_workers=_as_int(malfl_raw.get("num_workers", 0)),
    )

    # Optimization
    opt_raw = raw["optimization"]
    optimization = OptimizationConfig(
        policy_lr=_as_float(opt_raw["policy_lr"]),
        q_lr=_as_float(opt_raw["q_lr"]),
        value_regularizer=_as_float(opt_raw["value_regularizer"]),
    )

    # Logging
    log_raw = raw["logging"]
    logging = LoggingConfig(
        base_dir=str(log_raw["base_dir"]),
        save_trajectories=bool(log_raw["save_trajectories"]),
        save_policies=bool(log_raw["save_policies"]),
        save_models=bool(log_raw["save_models"]),
    )

    # Assemble experiment config
    return ExperimentConfig(
        alpha=_as_float(raw["alpha"]),
        gamma=_as_float(raw["gamma"]),
        horizon=_as_int(raw["horizon"]),
        device=str(raw.get("device", "cpu")),
        environment=env,
        maspi=maspi,
        malfl=malfl,
        optimization=optimization,
        logging=logging,
    )
