"""Simple discrete 3x3 GridWorld environment for MA-SPI sampling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np


Action = int
State = int


@dataclass
class TransitionResult:
    next_state: State
    reward: float
    terminated: bool


class GridWorld:
    """A minimal deterministic GridWorld with 3x3 grid and 5 actions."""

    def __init__(
        self,
        size: int = 3,
        max_steps: int = 20,
        start_state: State = 0,
        goal_state: State | None = None,
        step_reward: float = -0.01,
        goal_reward: float = 1.0,
    ) -> None:
        self.size = size
        self.n_states = size * size
        self.n_actions = 5  # up, down, left, right, stay
        self.start_state = start_state
        self.goal_state = goal_state if goal_state is not None else self.n_states - 1
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps

        self.state: State = self.start_state
        self.step_count: int = 0

        self.observation_space = gym.spaces.Discrete(self.n_states)
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def reset(self, *, seed: int | None = None) -> Tuple[State, dict]:
        if seed is not None:
            np.random.default_rng(seed)  # seed sink to mirror Gym API; env is deterministic
        self.state = self.start_state
        self.step_count = 0
        return self.state, {}

    def _index_to_rc(self, state: State) -> Tuple[int, int]:
        r = state // self.size
        c = state % self.size
        return r, c

    def _rc_to_index(self, row: int, col: int) -> State:
        return int(row * self.size + col)

    def transition_from(self, state: State, action: Action) -> TransitionResult:
        """Pure transition for planning/evaluation; does not mutate env state."""
        row, col = self._index_to_rc(state)
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # down
            row = min(self.size - 1, row + 1)
        elif action == 2:  # left
            col = max(0, col - 1)
        elif action == 3:  # right
            col = min(self.size - 1, col + 1)
        elif action == 4:  # stay
            pass
        else:
            raise ValueError(f"Invalid action {action}")

        next_state = self._rc_to_index(row, col)
        terminated = next_state == self.goal_state
        reward = self.goal_reward if terminated else self.step_reward
        return TransitionResult(next_state=next_state, reward=reward, terminated=terminated)

    def step(self, action: Action) -> Tuple[State, float, bool, bool, dict]:
        result = self.transition_from(self.state, action)
        self.state = result.next_state
        self.step_count += 1
        truncated = self.step_count >= self.max_steps and not result.terminated
        return result.next_state, result.reward, result.terminated, truncated, {}


__all__ = ["GridWorld", "TransitionResult", "Action", "State"]
