"""Run MA-LfL on T1 data and report metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from algorithms.baseline_lfl import run_baseline_lfl
from evaluation.metrics import compute_kl_errors, _gridworld_policy_from_reward, compute_weight_pcc, compute_weight_rmse
from envs.gridworld import GridWorld
from pathlib import Path
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MA-LfL baseline on T1 GridWorld")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (unused placeholder)")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    print(f"[run_ma_lfl] Config: {args.config}")

    result = run_baseline_lfl(cfg)
    env = GridWorld()
    # ground-truth reward: simple goal reward at terminal state
    true_w = np.zeros(result.feature_dim, dtype=float)
    true_w[env.n_states - 1] = 1.0  # reward on goal state indicator
    policy_true = _gridworld_policy_from_reward(true_w)
    policy_hat = _gridworld_policy_from_reward(result.w_hat)
    err_1a, err_1b = compute_kl_errors("gridworld", phases=[], policy_true=policy_true, policy_pred=policy_hat)
    diagnostics = {
        "weight_pcc": compute_weight_pcc(true_w, result.w_hat),
        "weight_rmse": compute_weight_rmse(true_w, result.w_hat),
    }

    metrics = {
        "env_name": "gridworld",
        "algorithm": "ma_lfl",
        "seed": args.seed,
        "err_1a": err_1a,
        "err_1b": err_1b,
        "diagnostics": diagnostics,
    }

    print(json.dumps({"err_1a": err_1a, "err_1b": err_1b, "feature_dim": result.feature_dim, "diagnostics": diagnostics}, indent=2))

    # Save metrics and weights for plotting/report
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"t1_ma_lfl_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[metrics] saved to {metrics_path}")

    weights_dir = Path("outputs/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.save(weights_dir / "t1_true.npy", true_w)
    np.save(weights_dir / "t1_hat.npy", result.w_hat)
    print(f"[weights] saved to {weights_dir}")


if __name__ == "__main__":
    main()
