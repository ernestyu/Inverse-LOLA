"""Run induced training on T2 using I-LOLA reward_hat.

This script:
1) Runs induced training for T2 (mpe_simple_spread) using a saved w_hat.
2) Saves results to outputs/induced/t2_induced_seed{seed}.json
3) Saves a plot to outputs/plots/t2_induced_returns_seed{seed}.png

It fixes a label/path bug where T2 was plotted with the T3 plotting function.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.induced_train import run_induced_training_t2  # noqa: E402


def _json_safe_number(x: Any) -> Optional[float]:
    """Convert to a JSON-safe float, using None for missing/NaN values."""
    if x is None:
        return None
    try:
        fx = float(x)
    except Exception:
        return None
    if math.isnan(fx) or math.isinf(fx):
        return None
    return fx


def _plot_induced_returns_t2(
    payload: Dict[str, Any],
    seed: int,
    out_path: Path,
) -> None:
    """Local plotting fallback to avoid wrong T3 labels and filenames."""
    import matplotlib.pyplot as plt  # local import to keep script lightweight

    r_induced_raw = payload.get("R_induced_curve", []) or []
    if r_induced_raw and isinstance(r_induced_raw[0], (list, tuple)) and len(r_induced_raw[0]) == 2:
        steps, r_induced = zip(*r_induced_raw)
    else:
        r_induced = list(r_induced_raw)
        steps = list(range(1, len(r_induced) + 1))

    plt.figure()
    if len(r_induced) > 0:
        plt.plot(steps, r_induced, label="R_induced")

    r_random = payload.get("R_random", None)
    if r_random is not None:
        plt.axhline(y=float(r_random), linestyle="--", label="R_random")

    r_expert = payload.get("R_expert", None)
    if r_expert is not None:
        plt.axhline(y=float(r_expert), linestyle="--", label="R_expert")

    plt.title(f"T2 MPE SimpleSpread induced returns (seed={seed})")
    plt.xlabel("training step")
    plt.ylabel("return")
    plt.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")


def _try_plot_with_repo_function(payload: Dict[str, Any], seed: int) -> bool:
    """Try to use a repo plotting function if it exists; otherwise fallback."""
    try:
        from evaluation import plots as repo_plots  # type: ignore
    except Exception:
        return False

    # Preferred: a dedicated T2 plotting function, if your repo has it.
    fn = getattr(repo_plots, "plot_t2_induced_returns", None)
    if callable(fn):
        fn(payload, seed=seed)
        return True

    # Second best: a generic plotting function, if your repo has it.
    fn = getattr(repo_plots, "plot_induced_returns", None)
    if callable(fn):
        fn(payload, seed=seed, task="t2")
        return True

    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/t2_mpe_simple_spread.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--w-hat",
        default=None,
        help="Path to w_hat; default uses outputs/weights/t2_ilola_seed{seed}.npy",
    )
    parser.add_argument(
        "--expert-ckpt",
        default=None,
        help="Optional expert checkpoint path. If not provided, R_expert will be None.",
    )
    args = parser.parse_args()

    if args.w_hat is None:
        args.w_hat = f"outputs/weights/t2_ilola_seed{args.seed}.npy"

    result = run_induced_training_t2(
        config_path=args.config,
        seed=args.seed,
        w_hat_path=args.w_hat,
        expert_ckpt_path=args.expert_ckpt,
    )

    payload: Dict[str, Any] = {
        "R_random": _json_safe_number(getattr(result, "R_random", None)),
        "R_expert": _json_safe_number(getattr(result, "R_expert", None)),
        "R_induced_curve": list(getattr(result, "R_induced_curve", []) or []),
        "data_seed": args.seed,
        "w_hat_path": args.w_hat,
    }

    out_dir = Path("outputs/induced")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"t2_induced_seed{args.seed}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[induced] saved {out_json}")

    # Plot: prefer repo plotting if it exists; otherwise use local T2 plot.
    used_repo_plot = _try_plot_with_repo_function(payload, seed=args.seed)
    if not used_repo_plot:
        out_plot = Path("outputs/plots") / f"t2_induced_returns_seed{args.seed}.png"
        _plot_induced_returns_t2(payload, seed=args.seed, out_path=out_plot)


if __name__ == "__main__":
    main()
