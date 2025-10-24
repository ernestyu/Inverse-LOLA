from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class RewardFamily(str, Enum):
    """Ground-truth reward families described in the MA-LfL paper."""

    HOMOGENEOUS = "homogeneous"
    HETEROGENEOUS = "heterogeneous"


Action = int
JointAction = Tuple[Action, Action]
Position = Tuple[int, int]
JointState = Tuple[Position, Position]


class GridWorld:
    """Deterministic 3x3 grid world used in MA-LfL experiments.

    The environment hosts two agents that move synchronously. Each agent starts
    from the top-left corner, and both share the same terminal goal at (2, 2).
    When an agent reaches the goal it is immediately teleported back to the
    start position, which mirrors the episodic reset described in the paper.
    """

    ACTIONS: Tuple[str, ...] = ("up", "down", "left", "right", "stay")
    ACTION_VECTORS: Tuple[Tuple[int, int], ...] = (
        (-1, 0),  # up
        (1, 0),  # down
        (0, -1),  # left
        (0, 1),  # right
        (0, 0),  # stay
    )

    def __init__(
        self,
        size: int = 3,
        start_positions: Tuple[Position, Position] = ((0, 0), (0, 0)),
        goal_position: Position = (2, 2),
        reward_family: RewardFamily = RewardFamily.HOMOGENEOUS,
        seed: int | None = None,
    ) -> None:
        if size != 3:
            raise ValueError("The engineering guide assumes a 3x3 grid.")

        self.size = size
        self.num_cells = size * size
        self.start_positions = (
            tuple(start_positions[0]),
            tuple(start_positions[1]),
        )
        self.goal_position = tuple(goal_position)
        self.reward_family = reward_family

        self._rng = np.random.default_rng(seed)
        self._state: JointState = (
            self.start_positions[0],
            self.start_positions[1],
        )

        self._positions: List[Position] = [
            (row, col) for row in range(size) for col in range(size)
        ]
        self._pos_to_index: Dict[Position, int] = {
            pos: idx for idx, pos in enumerate(self._positions)
        }
        self._joint_states: List[JointState] = [
            (p1, p2) for p1 in self._positions for p2 in self._positions
        ]

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #
    def reset(self) -> JointState:
        """Reset both agents to the start position."""
        self._state = (self.start_positions[0], self.start_positions[1])
        return self._state

    def step(self, actions: JointAction) -> Tuple[JointState, Tuple[float, float]]:
        """Apply a pair of actions and return the next joint state and rewards.

        Args:
            actions: Tuple containing the action index for each agent.
        """
        if len(actions) != 2:
            raise ValueError("Expected actions for two agents.")

        next_positions = []
        for idx, (position, action) in enumerate(zip(self._state, actions)):
            delta = self.ACTION_VECTORS[action]
            candidate = (position[0] + delta[0], position[1] + delta[1])
            clamped = (
                min(self.size - 1, max(0, candidate[0])),
                min(self.size - 1, max(0, candidate[1])),
            )
            if clamped == self.goal_position:
                next_positions.append(self.start_positions[idx])
            else:
                next_positions.append(clamped)

        next_state = (tuple(next_positions[0]), tuple(next_positions[1]))
        rewards = (
            self._reward(agent_index=0, state=self._state),
            self._reward(agent_index=1, state=self._state),
        )
        self._state = next_state
        return next_state, rewards

    def sample_joint_action(self) -> JointAction:
        """Sample a random joint action."""
        return (
            int(self._rng.integers(len(self.ACTIONS))),
            int(self._rng.integers(len(self.ACTIONS))),
        )

    def set_reward_family(self, reward_family: RewardFamily) -> None:
        self.reward_family = reward_family

    # --------------------------------------------------------------------- #
    # Utility methods                                                       #
    # --------------------------------------------------------------------- #
    @property
    def state(self) -> JointState:
        return self._state

    @property
    def state_index(self) -> int:
        return self.joint_state_to_index(self._state)

    def joint_state_to_index(self, state: JointState) -> int:
        return (
            self._pos_to_index[state[0]] * self.num_cells
            + self._pos_to_index[state[1]]
        )

    def index_to_joint_state(self, index: int) -> JointState:
        if index < 0 or index >= self.num_cells * self.num_cells:
            raise ValueError("Joint state index out of range.")
        first = index // self.num_cells
        second = index % self.num_cells
        return self._positions[first], self._positions[second]

    def all_joint_states(self) -> Iterable[JointState]:
        return tuple(self._joint_states)

    def all_state_indices(self) -> Iterable[int]:
        return range(self.num_cells * self.num_cells)

    def transition(self, state: JointState, actions: JointAction) -> JointState:
        """Deterministic transition used for expectation calculations."""
        current = self._state
        self._state = state
        next_state, _ = self.step(actions)
        self._state = current
        return next_state

    def reward(self, agent_index: int, state: JointState) -> float:
        """Return the ground-truth reward for the given joint state."""
        return self._reward(agent_index, state)

    def _reward(self, agent_index: int, state: JointState) -> float:
        pos_self = state[agent_index]
        pos_other = state[1 - agent_index]
        goal = self.goal_position

        dist_self_goal = manhattan_distance(pos_self, goal)
        dist_self_other = manhattan_distance(pos_self, pos_other)

        if self.reward_family == RewardFamily.HOMOGENEOUS:
            return -float(dist_self_goal) - float(dist_self_other)
        if self.reward_family == RewardFamily.HETEROGENEOUS:
            return -float(dist_self_goal) + float(dist_self_other)
        raise ValueError(f"Unsupported reward family: {self.reward_family}")


def manhattan_distance(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
