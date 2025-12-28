"""Induced training utilities for T2/T3 using reward_hat (minimal PPO)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from envs.mpe_simple_spread import make_env as make_env_t2
from envs.multiwalker import make_env as make_env_t3
from learners.ppo.mpe_runner import SharedPolicy as MPESharedPolicy
from learners.ppo.multiwalker_runner import SharedPolicy as MWSharedPolicy
from models.feature_maps import mpe_simple_features
from torch.nn.utils import vector_to_parameters

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - fallback if gym not present in minimal env
    gym = None


@dataclass
class InducedTrainingResult:
    R_random: float
    R_expert: float
    R_induced_curve: List[Tuple[int, float]]  # (update_idx, mean_return)


def eval_policy(env, policy: Any | None, episodes: int = 5, action_low=None, action_high=None) -> float:
    if episodes <= 0:
        print(f"[eval_policy] WARNING: num_episodes={episodes}, skipping evaluation.")
        return float("nan")
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            if policy is None:
                action = {aid: env.action_space(aid).sample() for aid in env.agents}
            else:
                action = {}
                for aid in env.agents:
                    obs_np = np.asarray(obs[aid], dtype=np.float32).reshape(-1)
                    obs_t = torch.as_tensor(obs_np)
                    if torch.isnan(obs_t).any():
                        obs_t = torch.nan_to_num(obs_t, nan=0.0)
                    with torch.no_grad():
                        # MultiWalker policy requires bounds; MPE policy takes only obs.
                        if action_low is None or action_high is None:
                            low = torch.as_tensor(env.action_space(aid).low, dtype=obs_t.dtype)
                            high = torch.as_tensor(env.action_space(aid).high, dtype=obs_t.dtype)
                        else:
                            low = action_low
                            high = action_high
                        device = next(policy.parameters()).device
                        obs_t = obs_t.to(device)
                        low = low.to(device)
                        high = high.to(device)
                        try:
                            act, _, _ = policy.act(obs_t)  # MPE SharedPolicy (obs only)
                        except TypeError:
                            act, _, _ = policy.act(obs_t, low, high)  # MultiWalker uses bounds
                        if torch.isnan(act).any():
                            act = torch.zeros_like(act)
                        act = torch.clamp(torch.nan_to_num(act, nan=0.0), low, high)
                    action[aid] = act.cpu().numpy()
            obs, rewards, term, trunc, _ = env.step(action)
            if rewards:
                ep_ret += float(np.mean(list(rewards.values())))
            done = all(term.values()) or all(trunc.values())
        rets.append(ep_ret)
    if not rets:
        print(f"[eval_policy] WARNING: no returns collected (episodes={episodes}).")
        return float("nan")
    mean_ret = float(np.mean(rets))
    print(f"[eval_policy] episodes={episodes}, mean_return={mean_ret}")
    return mean_ret


def _random_action_for_space(space):
    if gym and isinstance(space, gym.spaces.Discrete):
        return int(np.random.randint(space.n))
    if gym and isinstance(space, gym.spaces.MultiDiscrete):
        return space.sample()
    # default box/random sample
    if gym and isinstance(space, gym.spaces.Box):
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return np.random.uniform(low, high).astype(np.float32)
    return space.sample()


def eval_policy_random(env, episodes: int = 5) -> float:
    rets = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = {aid: _random_action_for_space(env.action_space(aid)) for aid in env.agents}
            obs, rewards, term, trunc, _ = env.step(action)
            if rewards:
                ep_ret += float(np.mean(list(rewards.values())))
            done = all(term.values()) or all(trunc.values())
        rets.append(ep_ret)
    if not rets:
        return float("nan")
    mean_ret = float(np.mean(rets))
    print(f"[eval_policy_random] episodes={episodes}, mean_return={mean_ret}")
    return mean_ret


def _set_policy_from_flat(policy: torch.nn.Module, theta_vec: np.ndarray) -> torch.nn.Module:
    flat = torch.as_tensor(np.asarray(theta_vec, dtype=np.float32)).flatten()
    total = sum(p.numel() for p in policy.parameters())
    if flat.numel() < total:
        pad = torch.zeros(total - flat.numel(), dtype=flat.dtype)
        flat = torch.cat([flat, pad], dim=0)
    elif flat.numel() > total:
        flat = flat[:total]
    vector_to_parameters(flat, policy.parameters())
    return policy


def _load_policy_from_ckpt(expert_ckpt_path: str, obs_dim: int, action_dim: int) -> MPESharedPolicy | None:
    policy = MPESharedPolicy(obs_dim, action_dim)
    ckpt = Path(expert_ckpt_path)
    if not ckpt.exists():
        return None
    try:
        if ckpt.suffix.lower() in {".pt", ".pth"}:
            state = torch.load(ckpt, map_location="cpu")
            try:
                policy.load_state_dict(state)
            except Exception:
                policy.load_state_dict(state.get("state_dict", state))
            return policy
        if ckpt.suffix.lower() in {".npy", ".npz"}:
            theta = np.load(ckpt)
            _set_policy_from_flat(policy, theta)
            return policy
    except Exception as e:
        print(f"[expert] failed to load ckpt {ckpt}: {e}")
    print(f"[expert] unsupported or failed ckpt {ckpt}; fallback to data.")
    return None


def _select_best_phase_expert(env, obs_dim: int, action_dim: int, eval_episodes: int = 20, seed: int | None = None) -> tuple[MPESharedPolicy | None, float | None]:
    try:
        from algorithms.i_logel import load_latest_t2  # local import to avoid cycle
    except Exception as e:
        print(f"[expert] cannot import load_latest_t2: {e}; expert unavailable.")
        return None, None

    try:
        _, phases = load_latest_t2(seed=seed)
    except Exception as e:
        print(f"[expert] failed to load T2 phases: {e}")
        return None, None
    if not phases:
        print("[expert] no T2 phases found; expert unavailable.")
        return None, None

    best_score: float | None = None
    best_policy: MPESharedPolicy | None = None
    for idx, phase in enumerate(phases):
        if not isinstance(phase.policy_params, dict):
            continue
        theta_vec = next(iter(phase.policy_params.values()))
        policy = MPESharedPolicy(obs_dim, action_dim)
        _set_policy_from_flat(policy, theta_vec)
        score = eval_policy(env, policy, episodes=eval_episodes)
        print(f"[expert] phase{idx} mean_return={score:.4f}")
        if (best_score is None) or (score > best_score):
            best_score = score
            best_policy = policy
    return best_policy, best_score


def _select_best_phase_expert_t3(env, obs_dim: int, action_dim: int, action_low: torch.Tensor, action_high: torch.Tensor, device: torch.device, eval_episodes: int = 20, seed: int | None = None) -> tuple[MWSharedPolicy | None, float | None]:
    try:
        from algorithms.i_lola import load_latest_t3  # local import to avoid cycle
    except Exception as e:
        print(f"[expert] cannot import load_latest_t3: {e}; expert unavailable.")
        return None, None

    try:
        _, phases = load_latest_t3(seed=seed)
    except Exception as e:
        print(f"[expert] failed to load T3 phases: {e}")
        return None, None
    if not phases:
        print("[expert] no T3 phases found; expert unavailable.")
        return None, None

    best_score: float | None = None
    best_policy: MWSharedPolicy | None = None
    for idx, phase in enumerate(phases):
        if not isinstance(phase.policy_params, dict):
            continue
        theta_vec = next(iter(phase.policy_params.values()))
        policy = MWSharedPolicy(obs_dim, action_dim).to(device)
        _set_policy_from_flat(policy, theta_vec)
        score = eval_policy(env, policy, episodes=eval_episodes, action_low=action_low, action_high=action_high)
        print(f"[expert-t3] phase{idx} mean_return={score:.4f}")
        if (best_score is None) or (score > best_score):
            best_score = score
            best_policy = policy
    return best_policy, best_score


def collect_rollout_with_reward_hat(env, policy: SharedPolicy, w_hat: np.ndarray, rollout_steps: int, gamma: float = 0.99):
    storage = []
    obs, _ = env.reset()
    k = len(w_hat)
    w_hat_t = torch.as_tensor(w_hat[:k], dtype=torch.float32)
    for _ in range(rollout_steps):
        actions = {}
        logps = {}
        values = {}
        for aid in env.agents:
            obs_np = np.asarray(obs[aid], dtype=np.float32)
            obs_t = torch.as_tensor(obs_np)
            if torch.isnan(obs_t).any():
                obs_t = torch.nan_to_num(obs_t, nan=0.0)
            act, logp, val = policy.act(obs_t)
            if torch.isnan(act).any():
                act = torch.zeros_like(act)
                logp = torch.zeros_like(logp)
                val = torch.zeros_like(val)
            low = torch.as_tensor(env.action_space(aid).low, dtype=act.dtype)
            high = torch.as_tensor(env.action_space(aid).high, dtype=act.dtype)
            act = torch.clamp(torch.nan_to_num(act, nan=0.0), low, high)
            actions[aid] = act.cpu().numpy()
            logps[aid] = logp
            values[aid] = val
        next_obs, rewards_env, term, trunc, _ = env.step(actions)

        # reward_hat per agent
        r_hats = {}
        for aid in env.agents:
            obs_np = np.asarray(obs[aid], dtype=np.float32)
            act_np = np.asarray(actions[aid], dtype=np.float32)
            phi = mpe_simple_features(obs_np, act_np)
            r_hats[aid] = float(torch.tensordot(phi, w_hat_t[: phi.numel()], dims=1).item())

        storage.append({
            "obs": obs,
            "actions": actions,
            "logps": logps,
            "values": values,
            "reward_hat": r_hats,
            "done": all(term.values()) or all(trunc.values()),
        })
        obs = next_obs
    return storage


def collect_rollout_with_reward_hat_t3(
    env,
    policy: MWSharedPolicy,
    w_hat: np.ndarray,
    rollout_steps: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    device: torch.device,
    gamma: float = 0.99,
):
    storage = []
    obs, _ = env.reset()
    w_hat_t = torch.as_tensor(w_hat, dtype=torch.float32, device=device)
    action_dim = action_low.numel()
    for _ in range(rollout_steps):
        actions = {}
        logps = {}
        values = {}
        for aid in env.agents:
            obs_np = np.asarray(obs[aid], dtype=np.float32).reshape(-1)
            obs_t = torch.as_tensor(obs_np, device=device)
            if torch.isnan(obs_t).any():
                obs_t = torch.nan_to_num(obs_t, nan=0.0)
            act, logp, val = policy.act(obs_t, action_low, action_high)
            if torch.isnan(act).any():
                act = torch.zeros_like(act)
                logp = torch.zeros_like(logp)
                val = torch.zeros_like(val)
            act = torch.clamp(torch.nan_to_num(act, nan=0.0), action_low, action_high)
            actions[aid] = act.cpu().numpy()
            logps[aid] = logp
            values[aid] = val
        next_obs, rewards_env, term, trunc, _ = env.step(actions)

        r_hats = {}
        for aid in env.agents:
            obs_np = np.asarray(obs[aid], dtype=np.float32)
            act_np = np.asarray(actions[aid], dtype=np.float32)
            phi = mpe_simple_features(obs_np, act_np, action_dim=action_dim)
            r_hats[aid] = float(torch.tensordot(phi.to(device), w_hat_t[: phi.numel()], dims=1).item())

        storage.append({
            "obs": obs,
            "actions": actions,
            "logps": logps,
            "values": values,
            "reward_hat": r_hats,
            "done": all(term.values()) or all(trunc.values()),
        })
        obs = next_obs
    return storage


def reinforce_update(policy: SharedPolicy, optimizer: torch.optim.Optimizer, storage: List[Dict[str, Any]], gamma: float = 0.99):
    # flatten agent trajectories; simple REINFORCE with mean reward across agents
    logps = []
    returns = []
    g = 0.0
    for step in reversed(storage):
        rewards_hat_vals = list(step["reward_hat"].values())
        # When the environment produces no reward_hat entries (e.g., empty agent list),
        # treat the reward as zero instead of calling np.mean([]) which triggers warnings.
        r_mean = float(np.mean(rewards_hat_vals)) if rewards_hat_vals else 0.0
        g = r_mean + gamma * g
        for lp in step["logps"].values():
            logps.append(lp)
            returns.append(g)
    if not logps:
        return
    logps_t = torch.stack(logps)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    loss = -(logps_t * returns_t).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def run_induced_training_t2(
    config_path: str,
    seed: int,
    w_hat_path: str,
    expert_ckpt_path: str | None = None,
) -> InducedTrainingResult:
    cfg = OmegaConf.load(config_path)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env_t2(num_agents=int(cfg.get("num_agents", 3)), max_cycles=int(cfg.get("max_cycles", 50)))

    w_hat = np.load(w_hat_path)
    # policy setup
    example_agent = env.agents[0]
    obs_dim = int(np.prod(env.observation_space(example_agent).shape))
    action_dim = int(np.prod(env.action_space(example_agent).shape))
    policy = MPESharedPolicy(obs_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg.get("lr", 3e-4)))

    eval_episodes = int(cfg.get("eval_episodes", 5))
    eval_episodes_expert = max(eval_episodes, int(cfg.get("eval_episodes_expert", 20)))

    R_random = eval_policy_random(env, episodes=eval_episodes_expert)

    expert_policy = None
    expert_score: float | None = None
    if expert_ckpt_path:
        expert_policy = _load_policy_from_ckpt(expert_ckpt_path, obs_dim, action_dim)
        if expert_policy is not None:
            expert_score = eval_policy(env, expert_policy, episodes=eval_episodes_expert)

    if expert_policy is None:
        expert_policy, expert_score = _select_best_phase_expert(env, obs_dim, action_dim, eval_episodes=eval_episodes_expert, seed=seed)

    R_expert = float(expert_score) if expert_score is not None else float("nan")

    num_updates = int(cfg.get("induced_updates", 5))
    rollout_steps = int(cfg.get("induced_rollout_steps", 64))
    eval_every = max(1, int(cfg.get("induced_eval_every", 1)))

    R_induced_curve: List[Tuple[int, float]] = []
    for update_idx in range(num_updates):
        storage = collect_rollout_with_reward_hat(env, policy, w_hat, rollout_steps)
        reinforce_update(policy, optimizer, storage)
        if (update_idx + 1) % eval_every == 0:
            R_ind = eval_policy(env, policy, episodes=eval_episodes_expert)
            R_induced_curve.append((update_idx + 1, R_ind))

    return InducedTrainingResult(R_random=R_random, R_expert=R_expert, R_induced_curve=R_induced_curve)


def run_induced_training_t3(
    config_path: str,
    seed: int,
    w_hat_path: str | None = None,
) -> InducedTrainingResult:
    cfg = OmegaConf.load(config_path)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(cfg.get("device", "cpu"))

    env = make_env_t3(
        num_agents=int(cfg.get("num_agents", 3)),
        max_cycles=int(cfg.get("max_cycles", 75)),
        seed=seed,
    )

    if w_hat_path is None:
        w_hat_path = Path("outputs") / "weights" / f"t3_ilola_seed{seed}.npy"
    w_hat_arr = np.load(w_hat_path)

    example_agent = env.agents[0]
    obs_dim = int(np.prod(env.observation_space(example_agent).shape))
    action_dim = int(np.prod(env.action_space(example_agent).shape))
    policy = MWSharedPolicy(obs_dim, action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(cfg.get("lr", 3e-4)))

    action_low = torch.as_tensor(env.action_space(example_agent).low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(env.action_space(example_agent).high, device=device, dtype=torch.float32)

    eval_episodes = int(cfg.get("eval_episodes", 5))
    eval_episodes_expert = max(eval_episodes, int(cfg.get("eval_episodes_expert", 20)))

    R_random = eval_policy(env, None, episodes=eval_episodes_expert, action_low=action_low, action_high=action_high)
    expert_policy, expert_score = _select_best_phase_expert_t3(
        env,
        obs_dim,
        action_dim,
        action_low=action_low,
        action_high=action_high,
        device=device,
        eval_episodes=eval_episodes_expert,
        seed=seed,
    )
    R_expert = float(expert_score) if expert_score is not None else float("nan")

    num_updates = int(cfg.get("induced_updates", 5))
    rollout_steps = int(cfg.get("induced_rollout_steps", 64))
    eval_every = max(1, int(cfg.get("induced_eval_every", 1)))

    R_induced_curve: List[Tuple[int, float]] = []
    for update_idx in range(num_updates):
        storage = collect_rollout_with_reward_hat_t3(
            env,
            policy,
            w_hat_arr,
            rollout_steps,
            action_low=action_low,
            action_high=action_high,
            device=device,
        )
        reinforce_update(policy, optimizer, storage)
        if (update_idx + 1) % eval_every == 0:
            R_ind = eval_policy(env, policy, episodes=eval_episodes_expert, action_low=action_low, action_high=action_high)
            R_induced_curve.append((update_idx + 1, R_ind))

    return InducedTrainingResult(R_random=R_random, R_expert=R_expert, R_induced_curve=R_induced_curve)


__all__ = ["InducedTrainingResult", "run_induced_training_t2", "run_induced_training_t3"]
