"""Run I-LOLA Stage B CMA-ES on T3 MultiWalker PPO data."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.i_lola import run_ilola_t3  # noqa: E402
from envs.multiwalker import make_env  # noqa: E402
from learners.ppo.multiwalker_runner import SharedPolicy  # noqa: E402
from evaluation.metrics import kl_mc_gaussian, collect_states_from_phase  # noqa: E402
from models.dynamics import step_lola  # noqa: E402
import json


def build_multiwalker_policy(env, device: torch.device):
    example_agent = env.agents[0]
    obs_dim = int(np.prod(env.observation_space(example_agent).shape))
    action_dim = int(np.prod(env.action_space(example_agent).shape))
    policy = SharedPolicy(obs_dim, action_dim).to(device)
    action_low = torch.as_tensor(env.action_space(example_agent).low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space(example_agent).high, device=device, dtype=torch.float32)
    return policy, action_low, action_high


def main() -> None:
    config_path = "configs/t3_multiwalker.yaml"
    cfg = OmegaConf.load(config_path)
    seed = 0
    alpha_scale = 0.5
    dim_limit = 16

    ilola_cfg = {
        "data_dir": "outputs/data/t3",
    }
    result, path, phases = run_ilola_t3(ilola_cfg, num_iters=5, alpha_scale=alpha_scale, dim_limit=dim_limit)
    print(f"[ilola-t3] data={path}")
    print(f"  CMA-ES iters={len(result.losses)}, final_loss={result.losses[-1]:.6f}")
    print(f"  omega_norm={np.linalg.norm(result.w_hat):.4f}")

    if len(phases) < 2:
        raise SystemExit("Need at least two phases to compute T3 KL.")
    p_t = phases[-2]
    p_tp1 = phases[-1]
    if not isinstance(p_t.policy_params, dict):
        raise SystemExit("Expected dict policy_params for T3 phases.")

    theta_t_full = next(iter(p_t.policy_params.values()))
    theta_tp1_full = next(iter(p_tp1.policy_params.values()))
    k = min(dim_limit, len(theta_t_full))
    theta_t = np.asarray(theta_t_full[:k], dtype=float)
    theta_tp1 = np.asarray(theta_tp1_full[:k], dtype=float)

    w_hat = np.asarray(result.w_hat, dtype=float).flatten()
    if w_hat.size < 2 * k:
        w_hat = np.concatenate([w_hat, np.zeros(2 * k - w_hat.size, dtype=float)])
    w_hat = w_hat[: 2 * k]
    w_a, w_b = w_hat[:k], w_hat[k : 2 * k]

    theta_pred_k, _ = step_lola(theta_t, theta_t, w_a, w_b, alpha_a=alpha_scale, alpha_b=alpha_scale)
    theta_pred_full = np.asarray(theta_t_full, dtype=float).copy()
    theta_pred_full[:k] = theta_pred_k
    diff_norm = np.linalg.norm(theta_tp1_full - theta_pred_full)
    print(f"[debug-t3] ||theta_true_next - theta_pred_next|| = {diff_norm}")

    env = make_env(
        num_agents=int(cfg.get("num_agents", 3)),
        max_cycles=int(cfg.get("max_cycles", 75)),
    )
    states = collect_states_from_phase(p_t, max_states=256, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    err_1b_raw = kl_mc_gaussian(
        env=env,
        build_policy_fn=build_multiwalker_policy,
        theta_p=np.asarray(theta_tp1_full, dtype=float),
        theta_q=theta_pred_full,
        states=states,
        num_action_samples=8,
        device=device,
    )
    err_1a = 0.0
    err_1b = max(float(err_1b_raw), 1e-12)
    print(f"  err_1a={err_1a:.6f}, err_1b={err_1b:.6f}")

    metrics = {
        "env_name": "multiwalker",
        "algorithm": "ilola",
        "seed": seed,
        "err_1a": err_1a,
        "err_1b": err_1b,
        "final_loss": float(result.losses[-1]),
        "omega_norm": float(np.linalg.norm(result.w_hat)),
    }
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"t3_ilola_seed{seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[metrics] saved to {metrics_path}")

    weights_dir = Path("outputs/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.save(weights_dir / f"t3_ilola_seed{seed}.npy", result.w_hat)
    print(f"[weights] saved to {weights_dir}")


if __name__ == "__main__":
    main()
