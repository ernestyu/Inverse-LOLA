"""Baseline MA-LfL runner for T1 GridWorld."""
from __future__ import annotations

import pickle
from pathlib import Path

from inverse.ma_lfl.core import ma_lfl_gridworld, MALfLOutput


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


def run_baseline_lfl(config: dict, seed: int | None = None) -> tuple[MALfLOutput, Path]:
    data_dir = Path(config.get("data_dir", "outputs/data/t1"))
    path, phases = load_latest_t1(data_dir, seed=seed)
    return ma_lfl_gridworld(phases), path


__all__ = ["run_baseline_lfl"]
