"""Simple PPO training loop on PettingZoo MultiWalker (or dummy fallback)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim

from data_gen.adapters import AgentRollout, PPORollout
from envs.multiwalker import make_env


def _init_weight(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("tanh"))
        nn.init.zeros_(m.bias)


class SharedPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        self.actor = nn.Sequential(*layers, nn.Linear(last, action_dim))
        self.critic = nn.Sequential(*layers, nn.Linear(last, 1))
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.apply(_init_weight)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def act(self, obs: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, action_low, action_high)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, value = self.forward(obs)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        clipped_actions = torch.clamp(actions, action_low, action_high)
        log_prob = dist.log_prob(clipped_actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


@dataclass
class RolloutBuffer:
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    log_probs: List[float]
    rewards: List[float]
    dones: List[bool]
    values: List[float]


def compute_gae(rewards: List[float], values: List[float], dones: List[bool], gamma: float, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
    advantages_np = np.array(advantages, dtype=np.float32)
    returns_np = advantages_np + np.array(values, dtype=np.float32)
    return advantages_np, returns_np


def collect_rollout(env, policy: SharedPolicy, steps: int, device: torch.device, gamma: float, gae_lambda: float):
    env_obs, _ = env.reset()
    active_agents = set(env.agents)
    buffers: Dict[str, RolloutBuffer] = {aid: RolloutBuffer([], [], [], [], [], []) for aid in active_agents}

    # action bounds from first agent (assume shared)
    sample_agent = next(iter(active_agents))
    act_space = env.action_space(sample_agent)
    action_low = torch.as_tensor(act_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(act_space.high, device=device, dtype=torch.float32)

    for _ in range(steps):
        actions = {}
        for aid in list(active_agents):
            obs_np = np.asarray(env_obs[aid], dtype=np.float32).reshape(-1)
            obs_tensor = torch.as_tensor(obs_np, device=device)
            with torch.no_grad():
                act, logp, value = policy.act(obs_tensor, action_low, action_high)
            actions[aid] = act.cpu().numpy()
            buffers[aid].observations.append(obs_np)
            buffers[aid].actions.append(actions[aid])
            buffers[aid].log_probs.append(float(logp.cpu().item()))
            buffers[aid].values.append(float(value.cpu().item()))

        env_obs, rewards, terminations, truncations, _ = env.step(actions)  # type: ignore[arg-type]

        finished_this_step = []
        for aid in list(active_agents):
            reward = float(rewards.get(aid, 0.0))
            done = bool(terminations.get(aid, False) or truncations.get(aid, False))
            buffers[aid].rewards.append(reward)
            buffers[aid].dones.append(done)
            if done:
                finished_this_step.append(aid)
        for aid in finished_this_step:
            active_agents.discard(aid)
        if not active_agents:
            break

    rollouts: Dict[str, AgentRollout] = {}
    adv_all = []
    ret_all = []
    obs_all = []
    act_all = []
    logp_all = []
    for aid, buf in buffers.items():
        adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, gamma=gamma, gae_lambda=gae_lambda)
        rollouts[aid] = AgentRollout(
            agent_id=aid,
            observations=buf.observations,
            actions=buf.actions,
            rewards=buf.rewards,
            dones=buf.dones,
            log_probs=buf.log_probs,
        )
        adv_all.append(adv)
        ret_all.append(ret)
        obs_all.append(np.array(buf.observations, dtype=np.float32))
        act_all.append(np.array(buf.actions, dtype=np.float32))
        logp_all.append(np.array(buf.log_probs, dtype=np.float32))

    obs_tensor = torch.as_tensor(np.concatenate(obs_all, axis=0), device=device)
    acts_tensor = torch.as_tensor(np.concatenate(act_all, axis=0), device=device)
    adv_tensor = torch.as_tensor(np.concatenate(adv_all, axis=0), device=device)
    ret_tensor = torch.as_tensor(np.concatenate(ret_all, axis=0), device=device)
    old_logp_tensor = torch.as_tensor(np.concatenate(logp_all, axis=0), device=device)
    return PPORollout(agent_rollouts=rollouts), obs_tensor, acts_tensor, adv_tensor, ret_tensor, old_logp_tensor, action_low, action_high


def ppo_update(
    policy: SharedPolicy,
    optimizer: optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    old_logp: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    clip_eps: float,
    epochs: int,
    minibatch_size: int,
    entropy_coef: float,
    vf_coef: float,
) -> dict:
    policy.train()
    losses = []
    for _ in range(epochs):
        idx = torch.randperm(obs.size(0), device=obs.device)
        for start in range(0, obs.size(0), minibatch_size):
            end = start + minibatch_size
            mb_idx = idx[start:end]
            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_adv = advantages[mb_idx]
            mb_ret = returns[mb_idx]
            mb_old_logp = old_logp[mb_idx]

            new_logp, entropy, values = policy.evaluate_actions(mb_obs, mb_actions, action_low, action_high)
            ratio = torch.exp(new_logp - mb_old_logp)
            adv_norm = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            surr1 = ratio * adv_norm
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_norm
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (values - mb_ret).pow(2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + vf_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            losses.append(
                {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.mean().item()),
                }
            )
    return {
        "policy_loss": np.mean([l["policy_loss"] for l in losses]),
        "value_loss": np.mean([l["value_loss"] for l in losses]),
        "entropy": np.mean([l["entropy"] for l in losses]),
    }


def run_ppo_training(config: dict, seed: int = 0, save_dir: Path | None = None):
    num_agents = int(config.get("num_agents", 3))
    max_cycles = int(config.get("max_cycles", 75))
    steps_per_rollout = int(config.get("steps_per_rollout", 48))
    updates = int(config.get("updates", 3))
    gamma = float(config.get("gamma", 0.95))
    gae_lambda = float(config.get("gae_lambda", 0.95))
    lr = float(config.get("lr", 3e-4))
    clip_eps = float(config.get("clip_eps", 0.2))
    entropy_coef = float(config.get("entropy_coef", 0.01))
    vf_coef = float(config.get("vf_coef", 0.5))
    minibatch_size = int(config.get("minibatch_size", 64))
    epochs = int(config.get("ppo_epochs", 3))

    env = make_env(num_agents=num_agents, max_cycles=max_cycles, seed=seed)
    example_agent = env.agents[0]
    obs_dim = int(np.prod(env.observation_space(example_agent).shape))
    action_dim = int(np.prod(env.action_space(example_agent).shape))

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SharedPolicy(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    raw_checkpoints = []
    save_dir = save_dir or Path("outputs") / "raw" / "t3"
    save_dir.mkdir(parents=True, exist_ok=True)

    for update_idx in range(updates):
        rollout, obs_tensor, act_tensor, adv_tensor, ret_tensor, old_logp_tensor, action_low, action_high = collect_rollout(
            env, policy, steps_per_rollout, device, gamma=gamma, gae_lambda=gae_lambda
        )
        stats = ppo_update(
            policy,
            optimizer,
            obs_tensor,
            act_tensor,
            adv_tensor,
            ret_tensor,
            old_logp_tensor,
            action_low=action_low,
            action_high=action_high,
            clip_eps=clip_eps,
            epochs=epochs,
            minibatch_size=minibatch_size,
            entropy_coef=entropy_coef,
            vf_coef=vf_coef,
        )
        theta_vec = nn.utils.parameters_to_vector(policy.parameters()).detach().cpu().numpy()
        ckpt = {"update_idx": update_idx, "theta": theta_vec, "rollouts": [rollout], "stats": stats}
        raw_checkpoints.append(ckpt)

        out_path = save_dir / f"ppo_multiwalker_update{update_idx}.pkl"
        with out_path.open("wb") as f:
            import pickle

            pickle.dump(ckpt, f)
        print(f"[ppo-multiwalker] update {update_idx}: saved {out_path}, stats={stats}")

    return raw_checkpoints


__all__ = ["run_ppo_training", "SharedPolicy"]
