"""Run MA-LfL mismatch evaluation on T2 PPO data and report KL errors."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.i_logel import load_latest_t2, run_ilogel_t2  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    align_reward_vector,
    collect_states_from_phase,
    compute_action_feature_means,
    compute_t2_ma_lfl_kl_errors,
)
from models.feature_maps import mpe_simple_features  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MA-LfL mismatch baseline on T2 PPO data")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling states")
    parser.add_argument("--alpha", type=float, default=0.5, help="Soft improvement mixing weight")
    parser.add_argument("--dim_limit", type=int, default=32, help="Max theta dims to keep for logits slice")
    parser.add_argument("--data_dir", type=str, default="outputs/data/t2", help="Directory with T2 PPO pkl data")
    parser.add_argument("--debug", action="store_true", help="Print debug info for KL inputs")
    parser.add_argument("--xphase", action="store_true", help="Enable cross-phase mismatch (phase0->1 fit, test on 1->2)")
    parser.add_argument("--holdout", type=float, default=0.0, help="Holdout ratio within a phase pair (train/test split). 0 to disable.")
    return parser.parse_args()


def extract_stage_a_reward(stage_a_result) -> np.ndarray | None:
    if hasattr(stage_a_result, "omega"):
        return np.asarray(stage_a_result.omega, dtype=float)
    if isinstance(stage_a_result, dict) and stage_a_result:
        first_val = next(iter(stage_a_result.values()))
        if hasattr(first_val, "omega"):
            return np.asarray(first_val.omega, dtype=float)
    return None


def recover_reward_mismatch(theta_t_full: np.ndarray, theta_tp1_full: np.ndarray, phi_means: np.ndarray, alpha: float, reg: float = 1e-3) -> np.ndarray:
    action_dim = phi_means.shape[0]
    theta_t = np.asarray(theta_t_full[:action_dim], dtype=float)
    theta_tp1 = np.asarray(theta_tp1_full[:action_dim], dtype=float)
    target = theta_tp1 - (1.0 - alpha) * theta_t
    M = alpha * phi_means  # shape: (action_dim, feature_dim)
    A = M.T @ M + reg * np.eye(M.shape[1])
    b = M.T @ target
    try:
        w_hat = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_hat = np.linalg.lstsq(A, b, rcond=None)[0]
    return w_hat


def proxy_reward_from_rollouts(phase, feature_dim: int, action_dim: int, reg: float = 1e-3) -> np.ndarray:
    feats: list[np.ndarray] = []
    rewards: list[float] = []
    for rollout in phase.trajectories:
        for ar in rollout.agent_rollouts.values():
            for obs, act, rew in zip(ar.observations, ar.actions, ar.rewards):
                act_vec = np.asarray(act, dtype=float).reshape(-1)
                if act_vec.size != action_dim:
                    if act_vec.size < action_dim:
                        pad = np.zeros(action_dim - act_vec.size, dtype=float)
                        act_vec = np.concatenate([act_vec, pad], axis=0)
                    else:
                        act_vec = act_vec[:action_dim]
                phi = mpe_simple_features(obs, act_vec, action_dim=action_dim)
                feats.append(phi.detach().cpu().numpy())
                rewards.append(float(rew))
    if not feats:
        return np.zeros(feature_dim, dtype=float)
    Phi = np.vstack(feats)
    r_vec = np.asarray(rewards, dtype=float)
    A = Phi.T @ Phi + reg * np.eye(feature_dim)
    b = Phi.T @ r_vec
    try:
        w_proxy = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w_proxy = np.linalg.lstsq(A, b, rcond=None)[0]
    if w_proxy.size > feature_dim:
        w_proxy = w_proxy[:feature_dim]
    elif w_proxy.size < feature_dim:
        w_proxy = np.concatenate([w_proxy, np.zeros(feature_dim - w_proxy.size, dtype=float)], axis=0)
    return w_proxy


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    path, phases = load_latest_t2(data_dir, seed=args.seed)
    path_seed = None
    for tok in path.stem.split("_"):
        if tok.startswith("seed") and tok[4:].isdigit():
            path_seed = int(tok[4:])
            break
    if path_seed is not None and path_seed != args.seed:
        raise SystemExit(f"[ma-lfl-mismatch] data seed mismatch: path has seed{path_seed}, expected seed{args.seed}")
    print(f"[ma-lfl-mismatch] data={path}")

    ilogel_cfg = {"data_dir": str(data_dir), "shared_policy": True}
    stage_a_result, _ = run_ilogel_t2(ilogel_cfg, gamma=0.95, num_iters=5)
    reward_true = extract_stage_a_reward(stage_a_result)

    if len(phases) < 2:
        raise SystemExit("Need at least two phases for mismatch evaluation.")

    # pick the phase pair with the largest parameter change to avoid degenerate near-zero steps (default mode)
    pair_norms: list[tuple[int, float]] = []
    for idx in range(len(phases) - 1):
        p_a = phases[idx]
        p_b = phases[idx + 1]
        if not isinstance(p_a.policy_params, dict):
            continue
        theta_a = next(iter(p_a.policy_params.values()))
        theta_b = next(iter(p_b.policy_params.values()))
        pair_norms.append((idx, float(np.linalg.norm(theta_b - theta_a))))
    if not pair_norms:
        raise SystemExit("No valid phase pairs found for mismatch evaluation.")
    best_idx, best_norm = max(pair_norms, key=lambda x: x[1])
    p_t = phases[best_idx]
    p_tp1 = phases[best_idx + 1]

    if not isinstance(p_t.policy_params, dict):
        raise SystemExit("Expected dict policy_params for PPO phases.")
    theta_t_full = next(iter(p_t.policy_params.values()))
    theta_tp1_full = next(iter(p_tp1.policy_params.values()))

    rollout = p_tp1.trajectories[0]
    aid = next(iter(rollout.agent_rollouts.keys()))
    action_dim = len(np.asarray(rollout.agent_rollouts[aid].actions[0]).reshape(-1))

    states = collect_states_from_phase(p_t, max_states=256, seed=args.seed)
    phi_means = compute_action_feature_means(states, action_dim)
    feature_dim = int(phi_means.shape[1])

    w_hat = recover_reward_mismatch(theta_t_full, theta_tp1_full, phi_means, alpha=args.alpha)
    if reward_true is None:
        reward_true = proxy_reward_from_rollouts(p_tp1, feature_dim=feature_dim, action_dim=action_dim)
        print("[ma-lfl-mismatch] Stage A reward unavailable; using proxy reward from env returns.")

    err_1a, err_1b, diagnostics = compute_t2_ma_lfl_kl_errors(
        phases=phases,
        reward_true=reward_true,
        reward_hat=w_hat,
        dim_limit=args.dim_limit,
        seed=args.seed,
        alpha=args.alpha,
        states=states,
        phi_means=phi_means,
        debug=args.debug,
    )
    print(f"  err_1a={err_1a:.6f}, err_1b={err_1b:.6f}")

    # Holdout within the selected pair (train/test split on states)
    err_1a_holdout = err_1b_holdout = None
    diagnostics_holdout: dict[str, float | list[int]] = {}
    if args.holdout and args.holdout > 0:
        if len(states) < 4:
            print("[ma-lfl-mismatch] not enough states for holdout split; skipping holdout.")
        else:
            rng = np.random.default_rng(args.seed)
            num_states = len(states)
            num_train = max(1, int((1.0 - args.holdout) * num_states))
            idx = rng.permutation(num_states)
            train_idx = idx[:num_train]
            test_idx = idx[num_train:]
            if len(test_idx) == 0:
                test_idx = idx[-1:]
            states_train = states[train_idx]
            states_test = states[test_idx]
            phi_train = compute_action_feature_means(states_train, action_dim)
            phi_test = compute_action_feature_means(states_test, action_dim)

            w_hat_holdout = recover_reward_mismatch(theta_t_full, theta_tp1_full, phi_train, alpha=args.alpha)
            err_1a_holdout, err_1b_holdout, diag_h = compute_t2_ma_lfl_kl_errors(
                phases=[p_t, p_tp1],
                reward_true=reward_true,
                reward_hat=w_hat_holdout,
                dim_limit=args.dim_limit,
                seed=args.seed,
                alpha=args.alpha,
                states=states_test,
                phi_means=phi_test,
                debug=args.debug,
            )
            print(f"  err_1a_holdout={err_1a_holdout:.6f}, err_1b_holdout={err_1b_holdout:.6f} (train={len(train_idx)}, test={len(test_idx)})")
            diagnostics_holdout = {
                "holdout_ratio": float(args.holdout),
                "holdout_train_size": int(len(train_idx)),
                "holdout_test_size": int(len(test_idx)),
                "w_hat_holdout_norm": float(np.linalg.norm(w_hat_holdout)),
                "phi_train_shape": list(phi_train.shape),
                "phi_test_shape": list(phi_test.shape),
                "max_diff_pi_real_hat_holdout": float(diag_h.get("max_diff_pi_real_hat", 0.0)),
                "max_diff_pi_real_true_holdout": float(diag_h.get("max_diff_pi_real_true", 0.0)),
            }

    # Cross-phase mismatch: fit on phase0->1, evaluate on phase1->2
    err_1a_x = err_1b_x = None
    diagnostics_x: dict[str, float | list[int] | None] = {}
    if args.xphase:
        if len(phases) < 3:
            print("[ma-lfl-mismatch] Need >=3 phases for cross-phase mode; skipping xphase.")
        else:
            p0, p1, p2 = phases[0], phases[1], phases[2]
            if not isinstance(p0.policy_params, dict) or not isinstance(p1.policy_params, dict) or not isinstance(p2.policy_params, dict):
                raise SystemExit("Expected dict policy_params for PPO phases in xphase mode.")
            theta0 = next(iter(p0.policy_params.values()))
            theta1 = next(iter(p1.policy_params.values()))
            theta2 = next(iter(p2.policy_params.values()))

            # train pair 0->1
            action_dim_train = len(np.asarray(p1.trajectories[0].agent_rollouts[next(iter(p1.trajectories[0].agent_rollouts))].actions[0]).reshape(-1))
            states_train = collect_states_from_phase(p0, max_states=256, seed=args.seed)
            phi_train = compute_action_feature_means(states_train, action_dim_train)
            w_hat_01 = recover_reward_mismatch(theta0, theta1, phi_train, alpha=args.alpha)

            # test pair 1->2
            action_dim_test = len(np.asarray(p2.trajectories[0].agent_rollouts[next(iter(p2.trajectories[0].agent_rollouts))].actions[0]).reshape(-1))
            states_test = collect_states_from_phase(p1, max_states=256, seed=args.seed)
            phi_test = compute_action_feature_means(states_test, action_dim_test)

            err_1a_x, err_1b_x, diag_x = compute_t2_ma_lfl_kl_errors(
                phases=[p1, p2],
                reward_true=reward_true,
                reward_hat=w_hat_01,
                dim_limit=args.dim_limit,
                seed=args.seed,
                alpha=args.alpha,
                states=states_test,
                phi_means=phi_test,
                debug=args.debug,
            )
            print(f"  err_1a_xphase={err_1a_x:.6f}, err_1b_xphase={err_1b_x:.6f}")
            diagnostics_x = {
                "xphase_train_pair": [0, 1],
                "xphase_test_pair": [1, 2],
                "w_hat_01_norm": float(np.linalg.norm(w_hat_01)),
                "phi_train_shape": list(phi_train.shape),
                "phi_test_shape": list(phi_test.shape),
                "max_diff_pi_real_hat_xphase": float(diag_x.get("max_diff_pi_real_hat", 0.0)),
                "max_diff_pi_real_true_xphase": float(diag_x.get("max_diff_pi_real_true", 0.0)),
            }

    metrics = {
        "env_name": "mpe_simple_spread",
        "algorithm": "ma_lfl",
        "mode": "mismatch",
        "seed": args.seed,
        "err_1a": err_1a,
        "err_1b": err_1b,
        "data_path": str(path),
        "data_seed_inferred": args.seed,
        "diagnostics": {
            **diagnostics,
            "best_pair_idx": int(best_idx),
            "best_pair_step_norm": float(best_norm),
            **diagnostics_x,
            **diagnostics_holdout,
        },
    }
    if err_1a_x is not None and err_1b_x is not None:
        metrics["err_1a_xphase"] = err_1a_x
        metrics["err_1b_xphase"] = err_1b_x
    if err_1a_holdout is not None and err_1b_holdout is not None:
        metrics["err_1a_holdout"] = err_1a_holdout
        metrics["err_1b_holdout"] = err_1b_holdout

    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"t2_ma_lfl_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[metrics] saved to {metrics_path}")

    weights_dir = Path("outputs/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    w_aligned = align_reward_vector(w_hat, feature_dim)
    np.save(weights_dir / f"t2_ma_lfl_seed{args.seed}.npy", w_aligned)
    print(f"[weights] saved to {weights_dir}")


if __name__ == "__main__":
    main()
