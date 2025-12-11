"""Stage B CMA-ES losses for LOLA simulation (single- and multi-step)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import cma

from models.dynamics import step_lola


@dataclass
class CMAESResult:
    w_hat: np.ndarray
    losses: list[float]


def one_step_loss(
    w_flat: np.ndarray,
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    w_dim: int,
    alpha_a: float,
    alpha_b: float,
    theta_a_next: np.ndarray,
    theta_b_next: np.ndarray,
) -> float:
    w_a = w_flat[:w_dim]
    w_b = w_flat[w_dim:]
    theta_a_pred, theta_b_pred = step_lola(theta_a, theta_b, w_a, w_b, alpha_a=alpha_a, alpha_b=alpha_b)
    loss = float(np.linalg.norm(theta_a_pred - theta_a_next) ** 2 + np.linalg.norm(theta_b_pred - theta_b_next) ** 2)
    return loss


def optimize_w_one_step(
    theta_a: np.ndarray,
    theta_b: np.ndarray,
    theta_a_next: np.ndarray,
    theta_b_next: np.ndarray,
    w_dim: int,
    alpha_a: float = 0.1,
    alpha_b: float = 0.1,
    init_w: np.ndarray | None = None,
    sigma: float = 0.5,
    maxiter: int = 30,
) -> CMAESResult:
    init_w = init_w if init_w is not None else np.zeros(2 * w_dim, dtype=float)
    es = cma.CMAEvolutionStrategy(init_w, sigma, {"maxiter": maxiter, "verb_disp": 0})
    losses: list[float] = []

    def f(w):
        val = one_step_loss(w, theta_a, theta_b, w_dim, alpha_a, alpha_b, theta_a_next, theta_b_next)
        losses.append(val)
        return val

    es.optimize(f)
    w_hat = np.array(es.result.xbest, dtype=float)
    return CMAESResult(w_hat=w_hat, losses=losses)


def multi_step_loss(
    w_flat: np.ndarray,
    thetas: Iterable[Tuple[np.ndarray, np.ndarray]],
    w_dim: int,
    alpha_a: float,
    alpha_b: float,
) -> float:
    """Accumulate squared errors across multiple (theta_t, theta_{t+1})."""
    w_a = w_flat[:w_dim]
    w_b = w_flat[w_dim:]
    total = 0.0
    for theta_a, theta_b, theta_a_next, theta_b_next in thetas:
        theta_a_pred, theta_b_pred = step_lola(theta_a, theta_b, w_a, w_b, alpha_a=alpha_a, alpha_b=alpha_b)
        total += np.linalg.norm(theta_a_pred - theta_a_next) ** 2 + np.linalg.norm(theta_b_pred - theta_b_next) ** 2
    return float(total)


def optimize_w_multistep(
    thetas: Iterable[Tuple[np.ndarray, np.ndarray]],
    w_dim: int,
    alpha_a: float = 0.1,
    alpha_b: float = 0.1,
    init_w: np.ndarray | None = None,
    sigma: float = 0.5,
    maxiter: int = 50,
) -> CMAESResult:
    thetas_list = list(thetas)
    init_w = init_w if init_w is not None else np.zeros(2 * w_dim, dtype=float)
    es = cma.CMAEvolutionStrategy(init_w, sigma, {"maxiter": maxiter, "verb_disp": 0})
    losses: list[float] = []

    def f(w):
        val = multi_step_loss(w, thetas_list, w_dim, alpha_a, alpha_b)
        losses.append(val)
        return val

    es.optimize(f)
    w_hat = np.array(es.result.xbest, dtype=float)
    return CMAESResult(w_hat=w_hat, losses=losses)


__all__ = ["one_step_loss", "optimize_w_one_step", "CMAESResult"]
