"""I-LOLA wrapper for CMA-ES recovery (single-step synthetic + multi-step T2)."""
from __future__ import annotations

import numpy as np

from inverse.ilola.stage_b_cmaes import (
    optimize_w_one_step,
    optimize_w_multistep,
    CMAESResult,
)
from algorithms.i_logel import load_latest_t2, run_ilogel_t2
from pathlib import Path
import pickle


def run_ilola_one_step_synthetic(theta_a, theta_b, theta_a_next, theta_b_next, w_dim: int, alpha_a=0.1, alpha_b=0.1) -> CMAESResult:
    init_w = np.zeros(2 * w_dim, dtype=float)
    return optimize_w_one_step(
        theta_a=np.asarray(theta_a, dtype=float),
        theta_b=np.asarray(theta_b, dtype=float),
        theta_a_next=np.asarray(theta_a_next, dtype=float),
        theta_b_next=np.asarray(theta_b_next, dtype=float),
        w_dim=w_dim,
        alpha_a=alpha_a,
        alpha_b=alpha_b,
        init_w=init_w,
    )


def run_ilola_t2(
    config: dict,
    mode: int = 1,
    gamma: float = 0.95,
    num_iters: int = 30,
    alpha_scale: float = 1.0,
    dim_limit: int = 64,
):
    """Run multi-step I-LOLA on T2 PPO data.

    mode 1: optimize W only (alphas fixed from Stage A if provided).
    mode 2: optimize W and scale alphas by a global lambda.
    """
    shared = bool(config.get("shared_policy", True))
    path, phases = load_latest_t2(Path(config.get("data_dir", "outputs/data/t2")))
    # Stage A estimates (shared or per-agent)
    stage_a_result, _ = run_ilogel_t2(config, gamma=gamma, num_iters=num_iters)

    # collect theta pairs from phases
    theta_pairs = []
    for phase_idx in range(len(phases) - 1):
        p_t = phases[phase_idx]
        p_tp1 = phases[phase_idx + 1]
        if not isinstance(p_t.policy_params, dict):
            continue
        theta_t_full = next(iter(p_t.policy_params.values()))
        theta_tp1_full = next(iter(p_tp1.policy_params.values()))
        k = min(dim_limit, len(theta_t_full))
        theta_t = theta_t_full[:k]
        theta_tp1 = theta_tp1_full[:k]
        theta_pairs.append((theta_t, theta_t, theta_tp1, theta_tp1))

    if not theta_pairs:
        raise SystemExit("No theta pairs collected for I-LOLA Stage B.")

    w_dim = len(theta_pairs[0][0])
    alpha_a = alpha_scale
    alpha_b = alpha_scale
    result = optimize_w_multistep(theta_pairs, w_dim=w_dim, alpha_a=alpha_a, alpha_b=alpha_b, maxiter=num_iters)
    return result, path, phases


def load_latest_t3(data_dir: Path | None = None):
    data_dir = data_dir or Path("outputs") / "data" / "t3"
    files = sorted(data_dir.glob("t3_ppo_seed*.pkl"))
    if not files:
        raise SystemExit(f"No T3 data found in {data_dir}. Run gen_data first.")
    latest = files[-1]
    with latest.open("rb") as f:
        phases = pickle.load(f)
    return latest, phases


def run_ilola_t3(
    config: dict,
    num_iters: int = 5,
    alpha_scale: float = 0.5,
    dim_limit: int = 16,
):
    path, phases = load_latest_t3(Path(config.get("data_dir", "outputs/data/t3")))

    theta_pairs = []
    for phase_idx in range(len(phases) - 1):
        p_t = phases[phase_idx]
        p_tp1 = phases[phase_idx + 1]
        if not isinstance(p_t.policy_params, dict):
            continue
        theta_t_full = next(iter(p_t.policy_params.values()))
        theta_tp1_full = next(iter(p_tp1.policy_params.values()))
        k = min(dim_limit, len(theta_t_full))
        theta_t = theta_t_full[:k]
        theta_tp1 = theta_tp1_full[:k]
        theta_pairs.append((theta_t, theta_t, theta_tp1, theta_tp1))
    if not theta_pairs:
        raise SystemExit("No theta pairs collected for I-LOLA Stage B on T3.")

    w_dim = len(theta_pairs[0][0])
    alpha_a = alpha_scale
    alpha_b = alpha_scale
    result = optimize_w_multistep(theta_pairs, w_dim=w_dim, alpha_a=alpha_a, alpha_b=alpha_b, maxiter=num_iters)
    return result, path, phases


__all__ = ["run_ilola_one_step_synthetic", "run_ilola_t2", "run_ilola_t3", "load_latest_t3"]
