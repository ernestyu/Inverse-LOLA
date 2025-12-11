"""Debug feature maps and reward nets on T1/T2 samples."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.feature_maps import (  # noqa: E402
    indicator_feature_gridworld,
    mpe_simple_features,
)
from models.reward_nets import LinearReward, RBFReward  # noqa: E402


def load_latest(path_glob: str):
    files = sorted(REPO_ROOT.glob(path_glob))
    if not files:
        raise SystemExit(f"No files match {path_glob}. Generate data first.")
    latest = files[-1]
    with latest.open("rb") as f:
        data = pickle.load(f)
    return latest, data


def debug_gridworld():
    path, phases = load_latest("outputs/data/t1/t1_ma_spi_seed*.pkl")
    phase0 = phases[0]
    traj0 = phase0.trajectories[0]
    s = traj0.states[0]
    a = traj0.actions[0]
    feat = indicator_feature_gridworld(s, a)
    lin = LinearReward(feat.shape[-1])
    reward = lin(feat).detach()
    print(f"[T1] path={path.name}, feature_dim={feat.numel()}, reward={reward.item():.4f}")


def debug_mpe():
    path, phases = load_latest("outputs/data/t2/t2_ppo_seed*.pkl")
    phase0 = phases[0]
    rollouts = phase0.trajectories
    agent_id = next(iter(rollouts[0].agent_rollouts.keys()))
    obs = rollouts[0].agent_rollouts[agent_id].observations[0]
    act = rollouts[0].agent_rollouts[agent_id].actions[0]
    feat = mpe_simple_features(obs, act, scale=5.0)
    lin = LinearReward(feat.shape[-1])
    reward = lin(feat).detach()
    print(f"[T2] path={path.name}, feat_dim={feat.numel()}, "
          f"feat_range=({feat.min().item():.3f},{feat.max().item():.3f}), reward={float(reward):.4f}")


def main() -> None:
    debug_gridworld()
    debug_mpe()


if __name__ == "__main__":
    main()
