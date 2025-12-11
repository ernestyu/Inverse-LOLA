"""Debug script for T1 MA-SPI data."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_latest_t1() -> tuple[Path, list]:
    data_dir = Path("outputs") / "data" / "t1"
    if not data_dir.exists():
        raise SystemExit(f"No data directory found at {data_dir}")
    candidates = sorted(data_dir.glob("t1_ma_spi_seed*.pkl"))
    if not candidates:
        raise SystemExit("No T1 MA-SPI data files found. Run runners.gen_data first.")
    latest = candidates[-1]
    with latest.open("rb") as f:
        phases = pickle.load(f)
    return latest, phases


def main() -> None:
    path, phases = load_latest_t1()
    print(f"[debug] Loaded {len(phases)} phases from {path}")
    for phase in phases[:3]:
        param_shape = getattr(phase, "policy_params", None)
        trajs = getattr(phase, "trajectories", [])
        shape_str = param_shape.shape if param_shape is not None else "unknown"
        print(f"  Phase {phase.phase_idx}: policy_params_shape={shape_str}, trajectories={len(trajs)}")
        if trajs:
            first_traj = trajs[0]
            print(f"    First traj length: {len(first_traj.actions)} steps, states_seen={len(first_traj.states)}")


if __name__ == "__main__":
    main()
