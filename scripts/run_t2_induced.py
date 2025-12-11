"""Run induced training on T2 using I-LOLA reward_hat."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.induced_train import run_induced_training_t2  # noqa: E402
from evaluation.plots import plot_t3_induced_returns  # noqa: E402


def main() -> None:
    config_path = "configs/t2_mpe_simple_spread.yaml"
    seed = 0
    w_hat_path = "outputs/weights/t2_ilola_seed0.npy"
    expert_ckpt_path = None  # placeholder; not loading expert for now

    result = run_induced_training_t2(
        config_path=config_path,
        seed=seed,
        w_hat_path=w_hat_path,
        expert_ckpt_path=expert_ckpt_path,
    )

    out_dir = Path("outputs/induced")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"t2_induced_seed{seed}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "R_random": result.R_random,
                "R_expert": result.R_expert,
                "R_induced_curve": result.R_induced_curve,
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
