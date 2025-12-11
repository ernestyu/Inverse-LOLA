"""Independent Stage A (I-LOGEL) alternating solver (single- and multi-agent)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from data_gen.adapters import LearningPhaseData, TabularPolicyModel, PPORollout


@dataclass
class ILogelResult:
    omega: np.ndarray
    alphas: np.ndarray
    losses: List[float]


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(z)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.clip(denom, 1e-9, None)


def _get_action_probs(phase: LearningPhaseData) -> np.ndarray:
    if isinstance(phase.policy_models, TabularPolicyModel):
        return np.asarray(phase.policy_models.action_probs, dtype=float)
    logits = np.asarray(phase.policy_params, dtype=float)
    return _softmax(logits)


def compute_reinforce_gradient(phase: LearningPhaseData, gamma: float = 0.99) -> np.ndarray:
    """Estimate grad_theta log pi weighted by returns using REINFORCE."""
    probs = _get_action_probs(phase)
    n_states, n_actions = probs.shape
    grad = np.zeros_like(probs, dtype=float)
    for traj in phase.trajectories:
        rewards = traj.rewards
        returns = []
        g = 0.0
        for r, done in zip(reversed(rewards), reversed(traj.dones)):
            g = r + gamma * g * (1.0 - float(done))
            returns.insert(0, g)
        returns = np.array(returns, dtype=float)
        for s, a, G in zip(traj.states[:-1], traj.actions, returns):
            pi_s = probs[s]
            one_hot = np.zeros_like(pi_s)
            one_hot[a] = 1.0
            grad[s] += G * (one_hot - pi_s)
    return grad.flatten()


def _discounted_return(rewards: List[float], gamma: float = 0.99) -> float:
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


def compute_multiagent_gradient(
    phase: LearningPhaseData,
    shared: bool = True,
    gamma: float = 0.99,
) -> Dict[str, np.ndarray] | np.ndarray:
    """Approximate gradient for PPO multi-agent phases using theta vectors and returns."""
    # theta params stored per-agent dict
    if not isinstance(phase.policy_params, dict):
        raise ValueError("Expected dict policy_params for multi-agent PPO data.")
    thetas: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in phase.policy_params.items()}
    # trajectories are PPORollout list
    rollouts: List[PPORollout] = phase.trajectories
    if shared:
        returns = []
        for ro in rollouts:
            for ar in ro.agent_rollouts.values():
                returns.append(_discounted_return(ar.rewards, gamma=gamma))
        mean_return = float(np.mean(returns)) if returns else 0.0
        # use first theta as representative
        theta_vec = next(iter(thetas.values()))
        return mean_return * theta_vec
    else:
        agent_grads: Dict[str, np.ndarray] = {}
        for ro in rollouts:
            for aid, ar in ro.agent_rollouts.items():
                g_ret = _discounted_return(ar.rewards, gamma=gamma)
                if aid not in agent_grads:
                    agent_grads[aid] = np.zeros_like(thetas[aid])
                agent_grads[aid] += g_ret * thetas[aid]
        # average over occurrences
        for aid in agent_grads:
            agent_grads[aid] = agent_grads[aid] / max(len(rollouts), 1)
        return agent_grads


def alternating_least_squares(grads: Iterable[np.ndarray], num_iters: int = 50, eps: float = 1e-6) -> ILogelResult:
    grads_list = [np.asarray(g, dtype=float) for g in grads]
    d = grads_list[0].shape[0]
    alphas = np.ones(len(grads_list), dtype=float)
    omega = np.mean(grads_list, axis=0)
    losses: List[float] = []
    for _ in range(num_iters):
        denom = np.sum(alphas ** 2) + eps
        omega = np.sum([a * g for a, g in zip(alphas, grads_list)], axis=0) / denom
        omega_norm2 = float(np.sum(omega ** 2)) + eps
        alphas = np.array([np.dot(omega, g) / omega_norm2 for g in grads_list], dtype=float)
        loss = float(np.mean([np.linalg.norm(g - a * omega) ** 2 for g, a in zip(grads_list, alphas)]))
        losses.append(loss)
    return ILogelResult(omega=omega, alphas=alphas, losses=losses)


def run_ilogel_stage_a(phases: Iterable[LearningPhaseData], gamma: float = 0.99, num_iters: int = 50) -> ILogelResult:
    grads = [compute_reinforce_gradient(phase, gamma=gamma) for phase in phases]
    return alternating_least_squares(grads, num_iters=num_iters)


def run_ilogel_stage_a_multi(
    phases: Iterable[LearningPhaseData],
    shared: bool = True,
    gamma: float = 0.99,
    num_iters: int = 50,
):
    if shared:
        grads = [compute_multiagent_gradient(p, shared=True, gamma=gamma) for p in phases]
        return alternating_least_squares(grads, num_iters=num_iters)
    else:
        # independent per-agent
        per_agent_grads: Dict[str, List[np.ndarray]] = {}
        for phase in phases:
            grad_dict = compute_multiagent_gradient(phase, shared=False, gamma=gamma)
            for aid, g in grad_dict.items():
                per_agent_grads.setdefault(aid, []).append(g)
        results: Dict[str, ILogelResult] = {}
        for aid, g_list in per_agent_grads.items():
            results[aid] = alternating_least_squares(g_list, num_iters=num_iters)
        return results


__all__ = [
    "compute_reinforce_gradient",
    "compute_multiagent_gradient",
    "alternating_least_squares",
    "run_ilogel_stage_a",
    "run_ilogel_stage_a_multi",
    "ILogelResult",
]
