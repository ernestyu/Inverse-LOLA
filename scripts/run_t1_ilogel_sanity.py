"""Sanity checks for I-LOGEL Stage A."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inverse.ilola.stage_a_independent import alternating_least_squares, run_ilogel_stage_a  # noqa: E402
from algorithms.i_logel import load_latest_t1  # noqa: E402


def synthetic_test():
    rng = np.random.default_rng(0)
    d = 12
    omega_true = rng.normal(scale=1.0, size=d)
    alphas_true = rng.uniform(0.5, 1.5, size=8)
    grads = [a * omega_true + 0.01 * rng.normal(size=d) for a in alphas_true]
    result = alternating_least_squares(grads, num_iters=50)
    cos_sim = float(np.dot(omega_true, result.omega) / (np.linalg.norm(omega_true) * np.linalg.norm(result.omega)))
    print("[synthetic] cos_sim(omega_true, omega_hat) = {:.4f}".format(cos_sim))
    print("[synthetic] alpha_true (first3) vs alpha_hat (first3):",
          list(alphas_true[:3]), list(result.alphas[:3]))


def real_data_test():
    path, phases = load_latest_t1()
    result = run_ilogel_stage_a(phases, gamma=0.95, num_iters=30)
    print(f"[real] data={path.name}, iters={len(result.losses)}, final_loss={result.losses[-1]:.6f}")
    print(f"[real] omega_norm={np.linalg.norm(result.omega):.4f}, alpha_mean={result.alphas.mean():.4f}")


def main() -> None:
    synthetic_test()
    real_data_test()


if __name__ == "__main__":
    main()
