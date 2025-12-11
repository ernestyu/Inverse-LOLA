"""Baseline MA-LfL runner for T1 GridWorld."""
from __future__ import annotations

import pickle
from pathlib import Path

from inverse.ma_lfl.core import ma_lfl_gridworld, MALfLOutput


def load_latest_t1(data_dir: Path | None = None):
    data_dir = data_dir or Path("outputs") / "data" / "t1"
    files = sorted(data_dir.glob("t1_ma_spi_seed*.pkl"))
    if not files:
        raise SystemExit(f"No T1 data found in {data_dir}. Run gen_data first.")
    latest = files[-1]
    with latest.open("rb") as f:
        phases = pickle.load(f)
    return latest, phases


def run_baseline_lfl(config: dict) -> MALfLOutput:
    data_dir = Path(config.get("data_dir", "outputs/data/t1"))
    _, phases = load_latest_t1(data_dir)
    return ma_lfl_gridworld(phases)


__all__ = ["run_baseline_lfl"]
