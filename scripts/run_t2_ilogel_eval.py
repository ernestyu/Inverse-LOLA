"""Run I-LOLA Stage B CMA-ES on T2 PPO data and report metrics."""
from __future__ import annotations

import sys
from pathlib import Path
import json

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.i_logel import run_ilogel_t2  # noqa: E402
from algorithms.i_lola import run_ilola_t2  # noqa: E402
from evaluation.metrics import compute_t2_ilola_kl_errors  # noqa: E402
import json
from pathlib import Path


def main() -> None:
    cfg = {
        "data_dir": "outputs/data/t2",
        "shared_policy": True,
    }
    stage_a_result, _ = run_ilogel_t2(cfg, gamma=0.95, num_iters=5)
    ilola_result, path, phases = run_ilola_t2(cfg, mode=1, gamma=0.95, num_iters=5, alpha_scale=0.5, dim_limit=32)
    # derive reference reward from Stage A (shared policy)
    if hasattr(stage_a_result, "omega"):
        reward_true = stage_a_result.omega
    elif isinstance(stage_a_result, dict):
        reward_true = next(iter(stage_a_result.values())).omega
    else:
        reward_true = None

    print(f"[ilola-t2] data={path}")
    print(f"  CMA-ES iters={len(ilola_result.losses)}, final_loss={ilola_result.losses[-1]:.6f}")
    print(f"  omega_norm={np.linalg.norm(ilola_result.w_hat):.4f}")
    # Debug norms to ensure rewards differ
    def _norm(x):
        import numpy as _np
        return float(_np.linalg.norm(_np.asarray(x).flatten())) if x is not None else -1.0
    print(f"[debug-t2] ||w_true|| = {_norm(reward_true):.6f}")
    print(f"[debug-t2] ||w_hat||  = {_norm(ilola_result.w_hat):.6f}")

    err_1a, err_1b = compute_t2_ilola_kl_errors(phases, reward_true=reward_true, reward_hat=ilola_result.w_hat, dim_limit=32, seed=0, alpha=0.5)
    print(f"  err_1a={err_1a:.6f}, err_1b={err_1b:.6f}")

    metrics = {
        "env_name": "mpe_simple_spread",
        "algorithm": "ilola",
        "seed": 0,
        "err_1a": err_1a,
        "err_1b": err_1b,
        "final_loss": float(ilola_result.losses[-1]),
        "omega_norm": float(np.linalg.norm(ilola_result.w_hat)),
    }
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"t2_ilola_seed0.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[metrics] saved to {metrics_path}")

    # Save w_hat for induced training
    weights_dir = Path("outputs/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.save(weights_dir / "t2_ilola_seed0.npy", ilola_result.w_hat)
    print(f"[weights] saved to {weights_dir}")


if __name__ == "__main__":
    main()
