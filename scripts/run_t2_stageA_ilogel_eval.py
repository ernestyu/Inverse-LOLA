"""Run I-LOGEL Stage A on T2 PPO data and report metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.i_logel import load_latest_t2, run_ilogel_t2  # noqa: E402
from evaluation.metrics import compute_t2_ilola_kl_errors  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A (I-LOGEL) evaluation on T2 PPO data")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount for Stage A")
    parser.add_argument("--num-iters", type=int, default=5, help="Stage A iterations")
    parser.add_argument("--dim-limit", type=int, default=32, help="Max theta dims")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {"data_dir": "outputs/data/t2", "shared_policy": True}

    stage_a_result, path_name = run_ilogel_t2(cfg, gamma=args.gamma, num_iters=args.num_iters, seed=args.seed)
    path, phases = load_latest_t2(Path(cfg["data_dir"]), seed=args.seed)
    path_seed = None
    for tok in Path(path).stem.split("_"):
        if tok.startswith("seed") and tok[4:].isdigit():
            path_seed = int(tok[4:])
            break
    if path_seed is not None and path_seed != args.seed:
        raise SystemExit(f"[stageA] data seed mismatch: path has seed{path_seed}, expected seed{args.seed}")
    reward_hat = np.asarray(stage_a_result.omega, dtype=float)

    err_1a, err_1b = compute_t2_ilola_kl_errors(
        phases=phases,
        reward_true=reward_hat,  # Stage A self-baseline
        reward_hat=reward_hat,
        dim_limit=args.dim_limit,
        seed=args.seed,
        alpha=0.5,
    )
    print(f"[stageA] data={path_name}")
    print(f"  err_1a={err_1a:.6f}, err_1b={err_1b:.6f}")

    metrics = {
        "env_name": "mpe_simple_spread",
        "algorithm": "ilogel_stageA",
        "seed": args.seed,
        "err_1a": err_1a,
        "err_1b": err_1b,
        "omega_norm": float(np.linalg.norm(reward_hat)),
        "num_iters": args.num_iters,
        "data_path": str(path),
        "data_seed_inferred": args.seed,
        "mode": "stageA",
    }
    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"t2_ilogel_stageA_seed{args.seed}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[metrics] saved to {metrics_path}")

    weights_dir = Path("outputs/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    np.save(weights_dir / f"t2_ilogel_stageA_seed{args.seed}.npy", reward_hat)
    print(f"[weights] saved to {weights_dir}")


if __name__ == "__main__":
    main()
