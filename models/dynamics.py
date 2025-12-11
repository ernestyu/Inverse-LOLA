"""Simple dynamics utilities including a minimal LOLA-style one-step update."""
from __future__ import annotations

import numpy as np
import torch


def step_lola(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    w_a: np.ndarray,
    w_b: np.ndarray,
    alpha_a: float = 0.1,
    alpha_b: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """One-step first-order LOLA-like update in parameter space.

    This is a lightweight simulator: treats theta as policy params, w as reward params,
    and uses simple linear rewards with gradient approximations.
    """
    theta_a_t = torch.as_tensor(theta_a, dtype=torch.float32)
    theta_b_t = torch.as_tensor(theta_b, dtype=torch.float32)
    w_a_t = torch.as_tensor(w_a, dtype=torch.float32)
    w_b_t = torch.as_tensor(w_b, dtype=torch.float32)

    # simple surrogate: gradient is outer product of opponent reward with own theta
    def grad_J(theta_self: torch.Tensor, theta_opp: torch.Tensor, w_self: torch.Tensor) -> torch.Tensor:
        return w_self * theta_self + 0.1 * theta_opp

    # update B first
    grad_b = grad_J(theta_b_t, theta_a_t, w_b_t)
    theta_b_new = theta_b_t + alpha_b * grad_b
    # then update A using updated B
    grad_a = grad_J(theta_a_t, theta_b_new, w_a_t)
    theta_a_new = theta_a_t + alpha_a * grad_a

    return theta_a_new.detach().numpy(), theta_b_new.detach().numpy()


__all__ = ["step_lola"]
