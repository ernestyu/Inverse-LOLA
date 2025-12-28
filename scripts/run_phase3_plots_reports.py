"""Generate plots and reports for Phase 3.2."""
from __future__ import annotations

import sys
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.plots import (  # noqa: E402
    plot_t1_kl,
    plot_t1_reward_heatmap,
    plot_t2_kl_compare,
    plot_t2_compare_lfl_ilola,
    plot_t2_stageA_vs_stageB,
    plot_t2_induced_returns,
    plot_t3_kl,
    plot_t3_induced_returns,
)
from evaluation.report import generate_report_t1, generate_report_t2, generate_report_t3  # noqa: E402


def main() -> None:
    plot_t1_kl(seed=0)
    plot_t1_reward_heatmap()
    plot_t2_kl_compare(seed=0)
    plot_t2_compare_lfl_ilola(seed=0)
    plot_t2_stageA_vs_stageB(seed=0)
    t2_induced = REPO_ROOT / "outputs" / "induced" / "t2_induced_seed0.json"
    if t2_induced.exists():
        with t2_induced.open("r", encoding="utf-8") as f:
            induced_result = json.load(f)
        plot_t2_induced_returns(induced_result, seed=0)
    else:
        print("[phase3] skip T2 induced plot (result not found).")

    generate_report_t1("configs/t1_gridworld.yaml", seed=0)
    generate_report_t2("configs/t2_mpe_simple_spread.yaml", seed=0)

    t3_metrics = REPO_ROOT / "outputs" / "metrics" / "t3_ilola_seed0.json"
    if t3_metrics.exists():
        plot_t3_kl(seed=0)
        generate_report_t3("configs/t3_multiwalker.yaml", seed=0)
    else:
        print("[phase3] skip T3 plot/report (metrics not found).")

    t3_induced = REPO_ROOT / "outputs" / "induced" / "t3_induced_seed0.json"
    if t3_induced.exists():
        with t3_induced.open("r", encoding="utf-8") as f:
            induced_result = json.load(f)
        plot_t3_induced_returns(induced_result, seed=0)
    else:
        print("[phase3] skip T3 induced plot (result not found).")


if __name__ == "__main__":
    main()
