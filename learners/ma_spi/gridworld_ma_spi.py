"""Simplified MA-SPI learner for the tabular GridWorld."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from envs.gridworld import GridWorld, TransitionResult


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-6)
    z = z - np.max(z, axis=-1, keepdims=True)
    exp = np.exp(z)
    denom = np.sum(exp, axis=-1, keepdims=True)
    return exp / np.clip(denom, 1e-9, None)


@dataclass
class TablePolicy:
    """Tabular softmax policy."""

    logits: np.ndarray
    temperature: float
    rng: np.random.Generator

    @classmethod
    def zeros(cls, num_states: int, num_actions: int, temperature: float, rng: np.random.Generator) -> "TablePolicy":
        logits = np.zeros((num_states, num_actions), dtype=float)
        return cls(logits=logits, temperature=temperature, rng=rng)

    def action_probs(self, state: int) -> np.ndarray:
        return _softmax(self.logits[state], self.temperature)

    def sample_action(self, state: int) -> int:
        probs = self.action_probs(state)
        return int(self.rng.choice(len(probs), p=probs))

    def copy(self) -> "TablePolicy":
        return TablePolicy(logits=self.logits.copy(), temperature=self.temperature, rng=self.rng)


class MASPIGridWorld:
    """Minimal MA-SPI loop with soft policy evaluation and improvement."""

    def __init__(
        self,
        env: GridWorld,
        gamma: float = 0.95,
        lr: float = 0.2,
        temperature: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.policy = TablePolicy.zeros(env.n_states, env.n_actions, temperature=temperature, rng=self.rng)

    def _state_action_value(self, state: int, action: int, value_fn: np.ndarray) -> float:
        transition: TransitionResult = self.env.transition_from(state, action)
        bootstrap = 0.0 if transition.terminated else value_fn[transition.next_state]
        return transition.reward + self.gamma * bootstrap

    def soft_policy_evaluation(self, policy: TablePolicy, iterations: int = 25, tol: float = 1e-4) -> np.ndarray:
        v = np.zeros(self.env.n_states, dtype=float)
        for _ in range(iterations):
            new_v = np.zeros_like(v)
            for s in range(self.env.n_states):
                probs = policy.action_probs(s)
                q_vals = [self._state_action_value(s, a, v) for a in range(self.env.n_actions)]
                new_v[s] = float(np.dot(probs, q_vals))
            if np.max(np.abs(new_v - v)) < tol:
                v = new_v
                break
            v = new_v
        return v

    def soft_policy_improvement(self, value_fn: np.ndarray) -> None:
        for s in range(self.env.n_states):
            q_vals = np.array([self._state_action_value(s, a, value_fn) for a in range(self.env.n_actions)], dtype=float)
            new_logits = q_vals / max(self.temperature, 1e-6)
            self.policy.logits[s] = (1 - self.lr) * self.policy.logits[s] + self.lr * new_logits

    def update_policy(self) -> np.ndarray:
        value_fn = self.soft_policy_evaluation(self.policy)
        self.soft_policy_improvement(value_fn)
        return value_fn

    def generate_trajectory(self, max_steps: int | None = None):
        """Roll out one episode using current policy."""
        states: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        dones: List[bool] = []

        obs, _ = self.env.reset()
        states.append(int(obs))
        horizon = max_steps or self.env.max_steps
        for _ in range(horizon):
            action = self.policy.sample_action(states[-1])
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            states.append(int(next_state))
            actions.append(int(action))
            rewards.append(float(reward))
            done = bool(terminated or truncated)
            dones.append(done)
            if done:
                break
        return states, actions, rewards, dones


__all__ = ["MASPIGridWorld", "TablePolicy"]
