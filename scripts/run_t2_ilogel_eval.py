"""Run I-LOLA Stage B CMA-ES on T2 PPO data and report metrics."""
from __future__ import annotations

import sys
from pathlib import Path
import json
import argparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.i_logel import run_ilogel_t2  # noqa: E402
from algorithms.i_lola import run_ilola_t2  # noqa: E402
from evaluation.metrics import compute_t2_ilola_kl_errors  # noqa: E402
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run I-LOLA Stage B on T2 PPO data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--alpha-scale", type=float, default=0.5)
    parser.add_argument("--dim-limit", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {
        "data_dir": "outputs/data/t2",
        "shared_policy": True,
    }
    stage_a_result, _ = run_ilogel_t2(cfg, gamma=args.gamma, num_iters=args.num_iters, seed=args.seed)
    ilola_result, path, phases = run_ilola_t2(
        cfg,
        mode=1,
        gamma=args.gamma,
        num_iters=args.num_iters,
        alpha_scale=args.alpha_scale,
        dim_limit=args.dim_limit,
        seed=args.seed,
    )
    path_seed = None
    for tok in Path(path).stem.split("_"):
        if tok.startswith("seed") and tok[4:].isdigit():
            path_seed = int(tok[4:])
            break
    if path_seed is not None and path_seed != args.seed:
        raise SystemExit(f"[ilola-t2] data seed mismatch: path has seed{path_seed}, expected seed{args.seed}")
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

    err_1a, err_1b = compute_t2_ilola_kl_errors(
        phases,
        reward_true=reward_true,
        reward_hat=ilola_result.w_hat,
        dim_limit=args.dim_limit,
        seed=args.seed,
        alpha=args.alpha_scale,
    )
    print(f"  err_1a={err_1a:.6f}, err_1b={err_1b:.6f}")

    metrics = {
        "env_name": "mpe_simple_spread",
        "algorithm": "ilola",
        "seed": args.seed,
        "err_1a": err_1a,
        "err_1b": err_1b,
        "final_loss": float(ilola_result.losses[-1]),
        "omega_norm": float(np.linalg.norm(ilola_result.w_hat)),
        "data_path": str(path),
        "data_seed_inferred": int(args.seed),
        "mode": "stageB",
    }
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"t2_ilola_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[metrics] saved to {metrics_path}")

    # Save w_hat for induced training
    weights_dir = Path("outputs/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.save(weights_dir / f"t2_ilola_seed{args.seed}.npy", ilola_result.w_hat)
    print(f"[weights] saved to {weights_dir}")


if __name__ == "__main__":
    main()
