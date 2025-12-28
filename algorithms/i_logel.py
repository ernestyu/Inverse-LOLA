"""High-level I-LOGEL interface for T1 GridWorld and T2 MPE."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

from inverse.ilola.stage_a_independent import run_ilogel_stage_a, run_ilogel_stage_a_multi, ILogelResult


def load_latest_t1(data_dir: Path | None = None, seed: int | None = None):
    data_dir = data_dir or Path("outputs") / "data" / "t1"
    pattern = f"t1_ma_spi_seed{seed}_*.pkl" if seed is not None else "t1_ma_spi_seed*.pkl"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No T1 data found in {data_dir} (seed={seed}). Run gen_data first.")
    latest = files[-1]
    with latest.open("rb") as f:
        phases = pickle.load(f)
    return latest, phases


def run_ilogel_t1(config: dict, gamma: float = 0.99, num_iters: int = 50, seed: int | None = None) -> Tuple[ILogelResult, str]:
    data_dir = Path(config.get("data_dir", "outputs/data/t1"))
    path, phases = load_latest_t1(data_dir, seed=seed)
    result = run_ilogel_stage_a(phases, gamma=gamma, num_iters=num_iters)
    return result, path.name


def load_latest_t2(data_dir: Path | None = None, seed: int | None = None):
    data_dir = data_dir or Path("outputs") / "data" / "t2"
    pattern = f"t2_ppo_seed{seed}_*.pkl" if seed is not None else "t2_ppo_seed*.pkl"
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No T2 data found in {data_dir} (seed={seed}). Run gen_data first.")
    latest = files[-1]
    with latest.open("rb") as f:
        phases = pickle.load(f)
    return latest, phases


def run_ilogel_t2(config: dict, gamma: float = 0.99, num_iters: int = 50, seed: int | None = None):
    data_dir = Path(config.get("data_dir", "outputs/data/t2"))
    shared = bool(config.get("shared_policy", True))
    path, phases = load_latest_t2(data_dir, seed=seed)
    result = run_ilogel_stage_a_multi(phases, shared=shared, gamma=gamma, num_iters=num_iters)
    return result, path.name


__all__ = ["run_ilogel_t1", "run_ilogel_t2"]
