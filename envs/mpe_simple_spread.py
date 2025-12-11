"""PettingZoo MPE simple_spread wrapper."""
from __future__ import annotations

from typing import Any

from pettingzoo.mpe import simple_spread_v3


def make_env(
    num_agents: int = 3,
    local_ratio: float = 0.5,
    max_cycles: int = 50,
    continuous_actions: bool = True,
    render_mode: str | None = None,
    seed: int | None = None,
) -> Any:
    """Create a parallel simple_spread environment."""
    env = simple_spread_v3.parallel_env(
        N=num_agents,
        local_ratio=local_ratio,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        render_mode=render_mode,
    )
    env.reset(seed=seed)
    return env


__all__ = ["make_env"]
