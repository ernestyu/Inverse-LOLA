"""Evaluation metrics for Project9 (KL, weight diagnostics, unified interface)."""
from __future__ import annotations

import numpy as np
import torch
from torch.nn.utils import vector_to_parameters

from envs.gridworld import GridWorld
from learners.ma_spi.gridworld_ma_spi import TablePolicy
from models.feature_maps import indicator_feature_gridworld, mpe_simple_features
from models.dynamics import step_lola
from data_gen.adapters import LearningPhaseData
from typing import Callable, Iterable

PolicyBuilder = Callable[[object, torch.device], tuple[torch.nn.Module, torch.Tensor, torch.Tensor]]


def _policy_kl_discrete(p_true: np.ndarray, p_pred: np.ndarray) -> float:
    p_true = np.clip(p_true, 1e-8, 1.0)
    p_pred = np.clip(p_pred, 1e-8, 1.0)
    return float(np.sum(p_true * (np.log(p_true) - np.log(p_pred))))


def compute_weight_pcc(w_true: np.ndarray, w_hat: np.ndarray) -> float:
    if w_true.size != w_hat.size:
        m = min(w_true.size, w_hat.size)
        w_true = w_true[:m]
        w_hat = w_hat[:m]
    return float(np.corrcoef(w_true, w_hat)[0, 1])


def compute_weight_rmse(w_true: np.ndarray, w_hat: np.ndarray) -> float:
    m = min(w_true.size, w_hat.size)
    return float(np.sqrt(np.mean((w_true[:m] - w_hat[:m]) ** 2)))


def _gridworld_policy_from_reward(reward: np.ndarray, temperature: float = 1.0) -> TablePolicy:
    env = GridWorld()
    rew = torch.as_tensor(reward, dtype=torch.float32)
    logits = torch.zeros((env.n_states, env.n_actions), dtype=torch.float32)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            logits[s, a] = rew.dot(indicator_feature_gridworld(s, a, n_states=env.n_states, n_actions=env.n_actions))
    return TablePolicy(logits=logits.numpy(), temperature=temperature, rng=np.random.default_rng(0))


def compute_kl_errors(
    env_name: str,
    phases,
    policy_true=None,
    policy_pred=None,
    reward_true=None,
    reward_hat=None,
    dim_limit: int = 32,
    num_samples: int = 256,
    seed: int = 0,
    alpha: float = 0.5,
):
    """Unified KL evaluation. Uses identical samples for 1a/1b to reduce variance.

    - For discrete actions (T1): exact sum over states.
    - For T2: LOLA-based lookahead with reward_true vs reward_hat.
    - For continuous (T3): still placeholder Monte Carlo if policies provided.
    """
    env_name_l = env_name.lower()
    if env_name_l in {"mpe_simple_spread", "simple_spread", "t2"} and reward_hat is not None:
        return compute_t2_ilola_kl_errors(
            phases=list(phases),
            reward_true=reward_true,
            reward_hat=reward_hat,
            dim_limit=dim_limit,
            seed=seed,
            alpha=alpha,
        )

    rng = np.random.default_rng(seed)
    if env_name_l == "gridworld":
        env = GridWorld()
        kls = []
        for s in range(env.n_states):
            p_true = policy_true.action_probs(s)
            p_pred = policy_pred.action_probs(s)
            kls.append(_policy_kl_discrete(p_true, p_pred))
        kl_1a = float(np.mean(kls))  # baseline/self-KL if policy_true==policy_pred
        kl_1b = kl_1a
        return kl_1a, kl_1b
    else:
        # Monte Carlo using recorded rollouts (agent_0)
        obs_list = []
        act_list = []
        for phase in phases:
            if not phase.trajectories:
                continue
            rollout = phase.trajectories[0]
            aid = next(iter(rollout.agent_rollouts.keys()))
            obs_list.extend(rollout.agent_rollouts[aid].observations)
            act_list.extend(rollout.agent_rollouts[aid].actions)
        if not obs_list:
            raise SystemExit("No observations found for KL computation.")
        idx = rng.choice(len(obs_list), size=min(num_samples, len(obs_list)), replace=False)
        kl_vals = []
        for i in idx:
            obs = np.asarray(obs_list[i], dtype=np.float32)
            act = np.asarray(act_list[i], dtype=np.float32)
            logp_true = policy_true.log_prob(obs, act)
            logp_pred = policy_pred.log_prob(obs, act)
            kl_vals.append(float(logp_true - logp_pred))
        kl_mean = float(np.mean(kl_vals))
        return kl_mean, kl_mean  # symmetric sampling; caller can pass different policies


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(), 1e-8, None)


def align_reward_vector(w_src: np.ndarray | None, target_dim: int) -> np.ndarray:
    """Pad or truncate reward vector to target_dim; return zeros if None."""
    if w_src is None:
        return np.zeros(target_dim, dtype=float)
    w_src = np.asarray(w_src, dtype=float).flatten()
    if w_src.size >= target_dim:
        return w_src[:target_dim]
    pad = np.zeros(target_dim - w_src.size, dtype=float)
    return np.concatenate([w_src, pad], axis=0)


def collect_states_from_phase(phase: LearningPhaseData, max_states: int = 256, seed: int = 0) -> np.ndarray:
    """Collect up to max_states flattened observations from a LearningPhaseData."""
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    for rollout in phase.trajectories or []:
        for ar in rollout.agent_rollouts.values():
            for obs in ar.observations:
                states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
    if not states:
        raise SystemExit("No observations available to compute KL.")
    states_np = np.stack(states, axis=0)
    if states_np.shape[0] > max_states:
        idx = rng.choice(states_np.shape[0], size=max_states, replace=False)
        states_np = states_np[idx]
    return states_np


def kl_mc_gaussian(
    env: object,
    build_policy_fn: PolicyBuilder,
    theta_p: np.ndarray,
    theta_q: np.ndarray,
    states: np.ndarray,
    num_action_samples: int = 8,
    device: torch.device | str = "cpu",
) -> float:
    """Monte Carlo KL for continuous actions using the PPO policy distribution.

    KL(pi_p || pi_q) = E_{s, a~pi_p}[log pi_p(a|s) - log pi_q(a|s)]
    """
    device_t = torch.device(device)
    policy_p, action_low, action_high = build_policy_fn(env, device_t)
    policy_q, _, _ = build_policy_fn(env, device_t)
    policy_p.eval()
    policy_q.eval()

    with torch.no_grad():
        def _align(theta_vec: np.ndarray, params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
            theta_np = np.asarray(theta_vec, dtype=np.float32).flatten()
            total = sum(p.numel() for p in params)
            if theta_np.size < total:
                pad = np.zeros(total - theta_np.size, dtype=np.float32)
                theta_np = np.concatenate([theta_np, pad], axis=0)
            else:
                theta_np = theta_np[:total]
            return torch.as_tensor(theta_np, device=device_t)

        vector_to_parameters(_align(theta_p, policy_p.parameters()), policy_p.parameters())
        vector_to_parameters(_align(theta_q, policy_q.parameters()), policy_q.parameters())

        obs = torch.as_tensor(states, dtype=torch.float32, device=device_t)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        N = obs.shape[0]
        kl_vals: list[float] = []
        log_std_p = policy_p.log_std.exp()
        log_std_q = policy_q.log_std.exp()
        for i in range(N):
            s = obs[i].unsqueeze(0).repeat(num_action_samples, 1)
            mean_p, _ = policy_p.forward(s)
            mean_q, _ = policy_q.forward(s)
            dist_p = torch.distributions.Normal(mean_p, log_std_p.expand_as(mean_p))
            dist_q = torch.distributions.Normal(mean_q, log_std_q.expand_as(mean_q))
            a = dist_p.sample()
            a_clipped = torch.clamp(a, action_low, action_high)
            logp = dist_p.log_prob(a_clipped).sum(-1)
            logq = dist_q.log_prob(a_clipped).sum(-1)
            kl_sample = float((logp - logq).mean().item())
            kl_vals.append(max(kl_sample, 0.0))
        return float(np.mean(kl_vals))


def compute_t2_ilola_kl_errors(
    phases: list[LearningPhaseData],
    reward_true: np.ndarray | None,
    reward_hat: np.ndarray,
    dim_limit: int = 32,
    seed: int = 0,
    alpha: float = 0.5,
) -> tuple[float, float]:
    """Simplified T2 KL using last phase pair and LOLA one-step prediction."""
    if len(phases) < 2:
        raise SystemExit("Need at least two phases for T2 KL computation.")
    rng = np.random.default_rng(seed)
    p_t = phases[-2]
    p_tp1 = phases[-1]
    if not isinstance(p_t.policy_params, dict):
        raise SystemExit("Expected dict policy_params for T2 phases.")
    theta_t_full = next(iter(p_t.policy_params.values()))
    theta_tp1_full = next(iter(p_tp1.policy_params.values()))
    k = min(dim_limit, len(theta_t_full))
    theta_t = np.asarray(theta_t_full[:k], dtype=float)
    theta_tp1 = np.asarray(theta_tp1_full[:k], dtype=float)

    # use first action dimension from trajectory action length
    rollout = p_tp1.trajectories[0]
    aid = next(iter(rollout.agent_rollouts.keys()))
    action_dim = len(rollout.agent_rollouts[aid].actions[0].reshape(-1))

    def align_w(w_src: np.ndarray | None) -> np.ndarray:
        if w_src is None:
            return np.zeros(2 * k, dtype=float)
        w_src = np.asarray(w_src, dtype=float).flatten()
        if len(w_src) >= 2 * k:
            return w_src[: 2 * k]
        pad = np.zeros(2 * k - len(w_src), dtype=float)
        return np.concatenate([w_src, pad], axis=0)

    w_true_aligned = align_w(reward_true)
    w_hat_aligned = align_w(reward_hat)

    def predict_theta(theta_base: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
        w_a = w_vec[:k]
        w_b = w_vec[k : 2 * k]
        theta_pred, _ = step_lola(theta_base, theta_base, w_a, w_b, alpha_a=alpha, alpha_b=alpha)
        return theta_pred

    theta_hat_true = predict_theta(theta_t, w_true_aligned)
    theta_hat_pred = predict_theta(theta_t, w_hat_aligned)

    pi_real = _softmax(theta_tp1[:action_dim])
    pi_ref = _softmax(theta_hat_true[:action_dim])
    pi_hat = _softmax(theta_hat_pred[:action_dim])

    obs_list = rollout.agent_rollouts[aid].observations
    idx = rng.choice(len(obs_list), size=min(len(obs_list), 64), replace=False)
    kl_list_1a = [_policy_kl_discrete(pi_real, pi_ref) for _ in idx]
    kl_list_1b = [_policy_kl_discrete(pi_real, pi_hat) for _ in idx]
    err_1a_raw = float(np.mean(kl_list_1a))
    err_1b_raw = float(np.mean(kl_list_1b))
    print("[debug-t2-metrics] err_1a_raw =", err_1a_raw)
    print("[debug-t2-metrics] err_1b_raw =", err_1b_raw)
    print("[debug-t2-metrics] kl_per_state_1a[:5] =", kl_list_1a[:5])
    print("[debug-t2-metrics] kl_per_state_1b[:5] =", kl_list_1b[:5])
    theta_diff = np.linalg.norm(theta_hat_true - theta_hat_pred)
    print("[debug-t2-metrics] ||theta_hat_true - theta_hat_pred|| =", theta_diff)
    err_1a = max(err_1a_raw, 1e-12)
    err_1b = max(err_1b_raw, 1e-12)
    return err_1a, err_1b


def compute_action_feature_means(states: np.ndarray, action_dim: int) -> np.ndarray:
    """Compute mean features for each discrete action placeholder in MPE-style PPO rollouts."""
    states_np = np.asarray(states, dtype=np.float32)
    actions = np.eye(action_dim, dtype=float)
    phi_means: list[np.ndarray] = []
    for act_vec in actions:
        feats = []
        for s in states_np:
            phi = mpe_simple_features(s, act_vec, action_dim=action_dim)
            feats.append(phi.detach().cpu().numpy())
        phi_means.append(np.mean(feats, axis=0))
    return np.stack(phi_means, axis=0)


def compute_t2_ma_lfl_kl_errors(
    phases: list[LearningPhaseData],
    reward_true: np.ndarray | None,
    reward_hat: np.ndarray,
    dim_limit: int = 32,
    seed: int = 0,
    alpha: float = 0.5,
    states: np.ndarray | None = None,
    phi_means: np.ndarray | None = None,
    debug: bool = False,
) -> tuple[float, float, dict]:
    """KL errors for MA-LfL mismatch on T2 PPO data using a soft-improvement lookahead."""
    if len(phases) < 2:
        raise SystemExit("Need at least two phases for T2 KL computation.")
    rng = np.random.default_rng(seed)
    p_t = phases[-2]
    p_tp1 = phases[-1]
    if not isinstance(p_t.policy_params, dict):
        raise SystemExit("Expected dict policy_params for T2 phases.")
    theta_t_full = next(iter(p_t.policy_params.values()))
    theta_tp1_full = next(iter(p_tp1.policy_params.values()))
    k = min(dim_limit, len(theta_t_full))

    rollout = p_tp1.trajectories[0]
    aid = next(iter(rollout.agent_rollouts.keys()))
    first_action = np.asarray(rollout.agent_rollouts[aid].actions[0]).reshape(-1)
    action_dim = min(len(first_action), k)

    theta_t = np.asarray(theta_t_full[:action_dim], dtype=float)
    theta_tp1 = np.asarray(theta_tp1_full[:action_dim], dtype=float)

    states_np = states if states is not None else collect_states_from_phase(p_t, max_states=256, seed=seed)
    phi_means_arr = phi_means if phi_means is not None else compute_action_feature_means(states_np, action_dim)
    feature_dim = int(phi_means_arr.shape[1])
    w_true_aligned = align_reward_vector(reward_true, feature_dim)
    w_hat_aligned = align_reward_vector(reward_hat, feature_dim)

    def _lfl_logits(theta_base: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
        q_vals = phi_means_arr @ w_vec
        blended = (1.0 - alpha) * theta_base + alpha * q_vals
        return blended

    logits_true = _lfl_logits(theta_t, w_true_aligned)
    logits_hat = _lfl_logits(theta_t, w_hat_aligned)

    pi_real = _softmax(theta_tp1)
    pi_ref = _softmax(logits_true)
    pi_hat = _softmax(logits_hat)

    if debug:
        max_diff_hat = float(np.max(np.abs(pi_real - pi_hat)))
        max_diff_true = float(np.max(np.abs(pi_real - pi_ref)))
        print("[debug-mismatch] pi_real[:5]      =", pi_real[:5])
        print("[debug-mismatch] pi_ref_true[:5]  =", pi_ref[:5])
        print("[debug-mismatch] pi_hat_hatW[:5]  =", pi_hat[:5])
        print("[debug-mismatch] max|real-hat|    =", max_diff_hat)
        print("[debug-mismatch] max|real-true|   =", max_diff_true)
        print("[debug-mismatch] shares_memory(real, hat)?", np.shares_memory(pi_real, pi_hat))
        print("[debug-mismatch] shares_memory(real, ref)?", np.shares_memory(pi_real, pi_ref))

    idx = rng.choice(len(states_np), size=min(len(states_np), 64), replace=False)
    kl_vals_1a = [_policy_kl_discrete(pi_real, pi_ref) for _ in idx]
    kl_vals_1b = [_policy_kl_discrete(pi_real, pi_hat) for _ in idx]
    err_1a = max(float(np.mean(kl_vals_1a)), 1e-12)
    err_1b = max(float(np.mean(kl_vals_1b)), 1e-12)
    diagnostics = {
        "action_dim": action_dim,
        "feature_dim": feature_dim,
        "w_true_norm": float(np.linalg.norm(w_true_aligned)),
        "w_hat_norm": float(np.linalg.norm(w_hat_aligned)),
        "theta_step_norm": float(np.linalg.norm(theta_tp1 - theta_t)),
        "phi_means_shape": list(phi_means_arr.shape),
        "max_diff_pi_real_hat": float(np.max(np.abs(pi_real - pi_hat))),
        "max_diff_pi_real_true": float(np.max(np.abs(pi_real - pi_ref))),
    }
    return err_1a, err_1b, diagnostics


__all__ = [
    "collect_states_from_phase",
    "compute_action_feature_means",
    "compute_kl_errors",
    "compute_t2_ma_lfl_kl_errors",
    "compute_weight_pcc",
    "compute_weight_rmse",
    "compute_t2_ilola_kl_errors",
    "align_reward_vector",
    "kl_mc_gaussian",
]
