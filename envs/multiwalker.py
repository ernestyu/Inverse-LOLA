"""PettingZoo MultiWalker wrapper with graceful fallback."""
from __future__ import annotations

from typing import Any

import numpy as np


def _make_dummy_env(num_agents: int = 3, obs_dim: int = 16, act_dim: int = 4, max_steps: int = 50):
    """A lightweight fallback parallel env if Box2D is unavailable."""
    import gymnasium as gym

    class DummyMultiWalker:
        metadata = {}

        def __init__(self) -> None:
            self.num_agents = num_agents
            self.agents = [f"agent_{i}" for i in range(num_agents)]
            self.possible_agents = list(self.agents)
            self._obs_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            self._act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            self.max_steps = max_steps
            self._step_count = 0

        def observation_space(self, agent: str):
            return self._obs_space

        def action_space(self, agent: str):
            return self._act_space

        def reset(self, seed: int | None = None):
            if seed is not None:
                np.random.seed(seed)
            self._step_count = 0
            self.agents = list(self.possible_agents)
            obs = {aid: self._obs_space.sample() * 0.0 for aid in self.agents}
            return obs, {}

        def step(self, actions: dict[str, np.ndarray]):
            self._step_count += 1
            obs = {aid: self._obs_space.sample() for aid in self.agents}
            rewards = {aid: float(np.random.uniform(-0.1, 1.0)) for aid in self.agents}
            done_flag = self._step_count >= self.max_steps
            terminations = {aid: done_flag for aid in self.agents}
            truncations = {aid: False for aid in self.agents}
            infos = {aid: {} for aid in self.agents}
            if done_flag:
                self.agents = []
            return obs, rewards, terminations, truncations, infos

    return DummyMultiWalker()


def make_env(
    num_agents: int = 3,
    max_cycles: int = 75,
    render_mode: str | None = None,
    seed: int | None = None,
) -> Any:
    """Create a parallel MultiWalker environment; falls back to dummy if Box2D missing."""
    try:
        from pettingzoo.sisl import multiwalker_v9
    except Exception:
        return _make_dummy_env(num_agents=num_agents, max_steps=max_cycles)

    env = multiwalker_v9.parallel_env(
        n_walkers=num_agents,
        position_noise=1e-3,
        angle_noise=1e-3,
        forward_reward=1.0,
        terminate_reward=-1.0,
        fall_reward=-10.0,
        terminate_on_fall=True,
        max_cycles=max_cycles,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


__all__ = ["make_env"]
