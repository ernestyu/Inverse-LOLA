"""Batch run all tasks for multiple seeds with strict error checking."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SEEDS = [0, 1, 2, 3, 4]
REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> None:
    full_cmd = [sys.executable] + cmd if cmd and cmd[0] != sys.executable else cmd
    print(f"[run] {' '.join(full_cmd)}")
    subprocess.run(full_cmd, check=True)


def main() -> None:
    for seed in SEEDS:
        print(f"\n===== SEED {seed} START =====")
        # T1
        run(["-m", "runners.gen_data", "--config", "configs/t1_gridworld.yaml", "--seed", str(seed)])
        run(["-m", "runners.run_ma_lfl", "--config", "configs/t1_gridworld.yaml", "--seed", str(seed)])
        # T2
        run(["-m", "runners.gen_data", "--config", "configs/t2_mpe_simple_spread.yaml", "--seed", str(seed)])
        run(["scripts/run_t2_stageA_ilogel_eval.py", "--seed", str(seed)])
        run(["scripts/run_t2_ilogel_eval.py", "--seed", str(seed)])
        run(["scripts/run_t2_ma_lfl_mismatch.py", "--seed", str(seed), "--xphase", "--holdout", "0.5"])
        run(["scripts/run_t2_induced.py", "--seed", str(seed)])
        # T3
        run(["-m", "runners.gen_data", "--config", "configs/t3_multiwalker.yaml", "--seed", str(seed)])
        run(["scripts/run_t3_ilola_eval.py", "--seed", str(seed)])
        run(["scripts/run_t3_induced.py", "--seed", str(seed)])
        print(f"===== SEED {seed} END =====")

    # final plots/reports (single call)
    run(["scripts/run_phase3_plots_reports.py"])


if __name__ == "__main__":
    main()
