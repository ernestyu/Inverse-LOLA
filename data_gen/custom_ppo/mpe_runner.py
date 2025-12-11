"""Wrapper to run PPO training for MPE and expose raw checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import List

from learners.ppo.mpe_runner import run_ppo_training


def generate_mpe_ppo_data(config: dict, seed: int = 0, save_dir: Path | None = None) -> List[dict]:
    return run_ppo_training(config, seed=seed, save_dir=save_dir)


__all__ = ["generate_mpe_ppo_data"]
