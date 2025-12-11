"""Debug script for T2 PPO data (MPE simple_spread)."""
from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_generation() -> None:
    cfg = "configs/t2_mpe_simple_spread.yaml"
    seed = 0
    try:
        subprocess.run(["make", "gen", f"CONF={cfg}", f"SEED={seed}"], check=True)
    except FileNotFoundError:
        # Fallback if make is unavailable on this platform.
        subprocess.run(
            [sys.executable, "-m", "runners.gen_data", "--config", cfg, "--seed", str(seed)],
            check=True,
        )


def load_latest_t2():
    data_dir = REPO_ROOT / "outputs" / "data" / "t2"
    if not data_dir.exists():
        raise SystemExit(f"No T2 data directory found at {data_dir}")
    files = sorted(data_dir.glob("t2_ppo_seed*.pkl"))
    if not files:
        raise SystemExit("No T2 PPO data files found. Run generation first.")
    latest = files[-1]
    with latest.open("rb") as f:
        phases = pickle.load(f)
    return latest, phases


def main() -> None:
    run_generation()
    path, phases = load_latest_t2()
    print(f"[debug] Loaded {len(phases)} PPO phases from {path}")
    for phase in phases:
        agents = list(phase.policy_params.keys()) if isinstance(phase.policy_params, dict) else []
        theta_len = len(next(iter(phase.policy_params.values()))) if agents else 0
        sample_counts = {aid: len(ar.actions) for aid, ar in phase.trajectories[0].agent_rollouts.items()} if phase.trajectories else {}
        print(f"  Phase {phase.phase_idx}: agents={agents}, theta_len={theta_len}, samples_per_agent={sample_counts}")


if __name__ == "__main__":
    main()
