"""Synthetic one-step I-LOLA + CMA-ES sanity check."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.i_lola import run_ilola_one_step_synthetic  # noqa: E402
from models.dynamics import step_lola  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    d = 10
    theta_a = rng.normal(size=d)
    theta_b = rng.normal(size=d)
    w_true = rng.normal(scale=0.5, size=2 * d)
    w_a_true = w_true[:d]
    w_b_true = w_true[d:]
    alpha_a = 0.1
    alpha_b = 0.1

    theta_a_next, theta_b_next = step_lola(theta_a, theta_b, w_a_true, w_b_true, alpha_a=alpha_a, alpha_b=alpha_b)

    result = run_ilola_one_step_synthetic(theta_a, theta_b, theta_a_next, theta_b_next, w_dim=d, alpha_a=alpha_a, alpha_b=alpha_b)
    est_w_a = result.w_hat[:d]
    est_w_b = result.w_hat[d:]

    print(f"[synthetic ilola] iterations={len(result.losses)}, final_loss={result.losses[-1]:.6f}")
    cos_a = np.dot(est_w_a, w_a_true) / (np.linalg.norm(est_w_a) * np.linalg.norm(w_a_true))
    cos_b = np.dot(est_w_b, w_b_true) / (np.linalg.norm(est_w_b) * np.linalg.norm(w_b_true))
    print(f"  cos_sim w_a: {cos_a:.4f}, w_b: {cos_b:.4f}")

    # Print a few loss values to show descent
    for i, l in enumerate(result.losses[:5]):
        print(f"  loss[{i}] = {l:.6f}")
    if len(result.losses) > 5:
        print(f"  loss[last] = {result.losses[-1]:.6f}")


if __name__ == "__main__":
    main()
