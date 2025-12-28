"""Run induced training on T3 MultiWalker using I-LOLA reward_hat."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.induced_train import run_induced_training_t3  # noqa: E402
from evaluation.plots import plot_t3_induced_returns  # noqa: E402
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run induced training on T3 MultiWalker")
    parser.add_argument("--config", type=str, default="configs/t3_multiwalker.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w-hat", type=str, default=None, help="Path to t3_ilola weights; defaults to outputs/weights/t3_ilola_seed{seed}.npy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    seed = args.seed
    w_hat_path = Path(args.w_hat) if args.w_hat else Path("outputs/weights") / f"t3_ilola_seed{seed}.npy"

    result = run_induced_training_t3(
        config_path=config_path,
        seed=seed,
        w_hat_path=str(w_hat_path),
    )

    out_dir = Path("outputs/induced")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"t3_induced_seed{seed}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "R_random": result.R_random,
                "R_expert": result.R_expert,
                "R_ppo_best": result.R_expert,
                "R_induced_curve": result.R_induced_curve,
                "data_seed": seed,
                "w_hat_path": str(w_hat_path),
            },
            f,
            indent=2,
        )
    print(f"[induced] saved {out_json}")

    plot_t3_induced_returns(
        {
            "R_random": result.R_random,
            "R_expert": result.R_expert,
            "R_induced_curve": result.R_induced_curve,
        },
        seed=seed,
    )


if __name__ == "__main__":
    main()
